#!/usr/bin/env python3
"""Run the real-runtime static/dynamic owner-interface probe.

This is the experiment-owned orchestration boundary for the frozen
``2026-08-05-static-dynamic-owner-interface-crossover`` unit.  The lower-level
modules intentionally contain no checkpoint loader; this runner binds their
seams to the HF session, the successful H0 trace, the native parser, and the
source panel.  The runner fails closed when any identity or actuator contract
is absent.  In particular, it never turns a teacher-forced score or a forced
row into a natural release result.

The module is also usable with a small injected adapter in CPU tests.  The
injected adapter is required to expose the same methods as the real adapter;
the test seam therefore exercises helper calls, parser calls, persistent hook
boundaries, and artifact collision without loading a model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Literal
from datetime import datetime, timezone

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_dynamic_history_and_crossover_probe as dynamic  # noqa: E402
from scripts.research import run_static_dynamic_gradient_path_audit as gradient  # noqa: E402
from scripts.research import run_static_post_llm_image_field_probe as static  # noqa: E402
from scripts.research import run_static_dynamic_owner_support_probe as support  # noqa: E402


UNIT_ID = "2026-08-05-static-dynamic-owner-interface-crossover"
SCHEMA_VERSION = "static_dynamic_owner_interface_experiment.v1"
HELPER_SCHEMA_VERSION = "helper_schema_bundle.v1"
RUNTIME_ATTESTATION_SCHEMA_VERSION = f"{SCHEMA_VERSION}.runtime_attestation.v1"
NOOP_TOLERANCE = 1e-4
ENDPOINT_SCHEMA_VERSION = "owner_interface.endpoint_receipt.v1"
NOT_MEASURED_STATUS = "not_measured"
MAX_NATIVE_ROW_TOKENS = 512
DEFAULT_MAX_RELEASE_TOKENS = 256
P1_ARM_IDS = (
    "K00",
    "K01",
    "K10",
    "K11",
    "K12",
    "K13",
    *(f"{arm}_block{layer}" for layer in (13, 23) for arm in ("R00", "R10", "R11", "R12")),
    "R00_block27",
    "R10_block27",
)
KNOWN_INELIGIBLE_PAIR_STATUSES = frozenset(
    {
        "no_verified_B",
        "indeterminate_image2299_support_transfer",
        "no_latest_covered_A",
    }
)
DEFAULT_PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/inputs/"
    "human-refined-13.geo_sorted_xy.coord.jsonl"
)
DEFAULT_COHORT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/cohort.json"
)
DEFAULT_H0_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/h0"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover"
)
PANEL_SOURCE_SHA256 = "01086b139fa23983697492fdb535b5154429277803e8f12b243f9a031d1451f8"
DERIVED_PANEL_SHA256 = "5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23"
ADAPTER_SHA256 = {
    "S": "49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da",
    "A": "b7fcb74a3dc7e8ed251b7b7513389164d1728358f4d99119e2335652711d21b7",
}
EMBEDDING_DELTA_SHA256 = {
    "S": "a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2",
    "A": "66ddb6f658340e69e94195dfb2e9f4a31f72f5536307749c640a6d30bc899eb5",
}

CHECKPOINTS: dict[str, dict[str, Any]] = {
    "S": {
        "config": Path(
            "/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/"
            "qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml"
        ),
        "wrapper": "object_box_closed",
        "h0_name": "s-step2444-h0",
    },
    "A": {
        "config": Path(
            "/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/"
            "qwen3_vl_2b_static_dynamic_owner_interface_a3_step2445_h0.yaml"
        ),
        "wrapper": "object_box_commit",
        "h0_name": "a3-step2445-h0",
    },
}


class OrchestrationError(RuntimeError):
    """A missing or incompatible real-runtime seam."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_token_ids(values: Sequence[int] | torch.Tensor) -> str:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().reshape(-1).tolist()
    return sha256_json([int(value) for value in values])


def sha256_tensor(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor hash requires a torch.Tensor")
    tensor = value.detach().cpu().contiguous()
    return sha256_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape), "values": tensor.tolist()})


def _not_measured(reason: str, **extra: Any) -> dict[str, Any]:
    """Represent unavailable diagnostic evidence without scientific nulls."""

    if not isinstance(reason, str) or not reason.strip():
        raise OrchestrationError("not_measured diagnostics require a non-empty reason")
    return {"status": NOT_MEASURED_STATUS, "reason": reason, **extra}


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OrchestrationError(f"{label} must be a finite numeric value")
    number = float(value)
    if not torch.isfinite(torch.tensor(number)).item():
        raise OrchestrationError(f"{label} must be a finite numeric value")
    return number


def _contiguous_runs(positions: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    values = tuple(sorted({int(position) for position in positions}))
    if not values:
        return ()
    runs: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value == runs[-1][-1] + 1:
            runs[-1].append(value)
        else:
            runs.append([value])
    return tuple(tuple(run) for run in runs)


def _native_region_view(hidden: torch.Tensor, positions: Sequence[int], label: str) -> torch.Tensor:
    """Select possibly fragmented positions while retaining one autograd graph."""

    runs = _contiguous_runs(positions)
    if not runs:
        raise OrchestrationError(f"P4 {label} position span is empty")
    pieces = [_native_arithmetic_position_view(hidden, run, label) for run in runs]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


def _bind_p4_logical_gradient_sources(
    hidden_by_role: Mapping[str, torch.Tensor],
    logical_positions: Mapping[str, Sequence[int]],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, tuple[gradient.GradientSource, ...]],
]:
    """Bind logical P4 regions to native block outputs without derived copies.

    Fragmented image regions cannot be represented by one storage-sharing 2-D
    view: concatenating their contiguous runs creates a new tensor that is no
    longer the block-23 hook output.  P4 already differentiates the complete
    per-candidate hook tensors and slices logical positions afterward through
    ``GradientSource``.  Retain those native tensors as the captured states as
    well, while the position-bearing sources remain the sole region selector.
    """

    if not hidden_by_role:
        raise OrchestrationError("P4 requires at least one candidate block23 output")
    normalized_hidden: dict[str, torch.Tensor] = {}
    for role, hidden in hidden_by_role.items():
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
            raise OrchestrationError(f"P4 {role} block23 output must have shape [1,S,H]")
        normalized_hidden[str(role)] = hidden
    anchor = next(iter(normalized_hidden.values()))
    states: dict[str, torch.Tensor] = {}
    sources: dict[str, tuple[gradient.GradientSource, ...]] = {}
    for raw_name, raw_positions in logical_positions.items():
        name = str(raw_name)
        positions = tuple(int(position) for position in raw_positions)
        if not positions:
            raise OrchestrationError(f"P4 {name} logical position span is empty")
        for role, hidden in normalized_hidden.items():
            if any(position < 0 or position >= int(hidden.shape[1]) for position in positions):
                raise OrchestrationError(
                    f"P4 {name} logical position exceeds {role} diagnostic forward"
                )
        states[name] = anchor
        sources[name] = tuple(
            gradient.GradientSource(hidden, positions, role)
            for role, hidden in normalized_hidden.items()
        )
    return states, sources


def _segment_token_indices(
    row_token_ids: Sequence[int], *, contract: static.WrapperContract
) -> dict[str, tuple[int, ...]]:
    """Return exact full-row token indices for description and XYXY segments."""

    row = [int(token) for token in row_token_ids]
    if not row or row[0] != contract.object_ref_start_token_id:
        raise OrchestrationError("endpoint row must begin with the native object-ref opener")
    if row.count(contract.object_ref_end_token_id) != 1:
        raise OrchestrationError("endpoint row must contain one object-ref closer")
    object_end = row.index(contract.object_ref_end_token_id)
    if object_end <= 1:
        raise OrchestrationError("endpoint row description segment is empty")
    if object_end + 1 >= len(row) or row[object_end + 1] != contract.box_start_token_id:
        raise OrchestrationError("endpoint row box-start marker is not adjacent to description")
    coord_start = object_end + 2
    coord_end = coord_start + contract.coordinate_count
    if coord_end >= len(row):
        raise OrchestrationError("endpoint row coordinate segment is truncated")
    coords = row[coord_start:coord_end]
    coordinate_min = contract.coordinate_token_start_id
    coordinate_max = coordinate_min + contract.coordinate_bin_count
    if any(not coordinate_min <= token < coordinate_max for token in coords):
        raise OrchestrationError("endpoint row coordinate segment has wrong tokenization")
    if row[coord_end] != contract.box_end_token_id:
        raise OrchestrationError("endpoint row lacks native box closure")
    if contract.commit_token_id is None:
        if coord_end != len(row) - 1:
            raise OrchestrationError("closed endpoint row has over-continuation after box closure")
        closure = (coord_end,)
    else:
        if coord_end + 1 >= len(row) or row[coord_end + 1] != contract.commit_token_id:
            raise OrchestrationError("commit endpoint row lacks native commit closure")
        if row.count(contract.commit_token_id) != 1 or coord_end + 1 != len(row) - 1:
            raise OrchestrationError("commit endpoint row has over-continuation")
        closure = (coord_end, coord_end + 1)
    return {
        "row_entry": (0,),
        "description": tuple(range(1, object_end)),
        "geometry": tuple(range(object_end + 1, coord_end)),
        "x1": (coord_start,),
        "y1": (coord_start + 1,),
        "x2": (coord_start + 2,),
        "y2": (coord_start + 3,),
        "closure": closure,
        "full_row": tuple(range(len(row))),
    }


def _natural_segment_scores(
    row_token_ids: Sequence[int],
    selected_log_probs: Sequence[float] | None,
    selected_ranks: Sequence[int] | None,
    *,
    contract: static.WrapperContract,
) -> dict[str, Any]:
    """Summarize selected natural-token scores with checked segment arithmetic."""

    try:
        phases = _segment_token_indices(row_token_ids, contract=contract)
    except OrchestrationError as exc:
        return _not_measured(f"natural row segment arithmetic failed: {exc}")
    logs = tuple(selected_log_probs or ())
    ranks = tuple(selected_ranks or ())
    # The row opener is already the last token of the exact natural prefix, so
    # generated selected-token arrays align to full-row positions [1:].
    if len(logs) != len(row_token_ids) - 1 or (ranks and len(ranks) != len(logs)):
        return _not_measured(
            "selected natural token scores do not align to the native row suffix",
            expected_suffix_tokens=len(row_token_ids) - 1,
            observed_logprob_tokens=len(logs),
            observed_rank_tokens=len(ranks),
        )
    try:
        values = tuple(_finite_number(value, label="selected token log-probability") for value in logs)
    except OrchestrationError as exc:
        return _not_measured(str(exc))
    result: dict[str, Any] = {
        "status": "measured",
        "source": "natural_selected_token_logprobs",
        "teacher_forced": False,
        "segments": {},
    }
    for name, indices in phases.items():
        suffix_indices = tuple(index - 1 for index in indices if index > 0)
        if name == "row_entry":
            # The row opener is scored at the previous boundary and therefore
            # intentionally remains separate from this suffix-only summary.
            continue
        if not suffix_indices:
            result["segments"][name] = _not_measured("segment has no generated suffix tokens")
            continue
        segment_values = [values[index] for index in suffix_indices]
        segment_ranks = [int(ranks[index]) for index in suffix_indices] if ranks else None
        if segment_ranks is not None and any(rank <= 0 for rank in segment_ranks):
            result["segments"][name] = _not_measured("selected token rank is non-positive")
            continue
        result["segments"][name] = {
            "status": "measured",
            "sum_log_probability": float(sum(segment_values)),
            "mean_log_probability": float(sum(segment_values) / len(segment_values)),
            "token_count": len(segment_values),
            "token_indices": list(indices),
            "selected_token_log_probabilities": segment_values,
            "selected_token_ranks": segment_ranks,
        }
    result["margins"] = _not_measured(
        "no competing candidate was scored in the identical natural release forward"
    )
    return result


def _score_active_endpoint_candidates(
    adapter: Any,
    context: EventContext,
    *,
    active_intervention: str,
    forward_fn: Callable[[torch.Tensor], tuple[Any, Mapping[str, Any]]],
    contract: static.WrapperContract,
) -> dict[str, Any]:
    """Score endpoint candidates with the active arm, one exact scalar forward at a time.

    The natural release remains the primary endpoint.  This helper only adds a
    teacher-forced diagnostic: each candidate starts from the same exact native
    prefix, receives the active K/R or D/Y operator at every scalar boundary,
    and is never scored from the natural-release token log-probabilities.  A
    callback receipt is required for every forward so an intervention arm
    cannot silently fall back to native/no-op logits.
    """

    def output_logits(output: Any) -> torch.Tensor:
        raw = (
            output.logits
            if hasattr(output, "logits")
            else output.get("logits")
            if isinstance(output, Mapping)
            else None
        )
        if not isinstance(raw, torch.Tensor) or raw.ndim != 3 or raw.shape[0] != 1:
            raise OrchestrationError(
                "active endpoint scalar forward did not expose [1,sequence,vocab] logits"
            )
        return raw[0, -1].detach().float()

    def active_logits(ids: torch.Tensor) -> tuple[torch.Tensor, Mapping[str, Any]]:
        output, receipt = forward_fn(ids)
        if not isinstance(receipt, Mapping):
            raise OrchestrationError("active endpoint scalar forward lacks an intervention receipt")
        observed = str(receipt.get("active_intervention", ""))
        if observed != str(active_intervention):
            raise OrchestrationError(
                f"active endpoint scalar receipt binds {observed!r}, expected {active_intervention!r}"
            )
        if str(active_intervention) not in {"K00", "D00", "Y00"} and receipt.get("intervention_applied") is not True:
            raise OrchestrationError(
                "active endpoint scalar receipt does not prove the requested intervention was applied"
            )
        logits = output_logits(output)
        if not bool(torch.isfinite(logits).all().item()):
            raise OrchestrationError("active endpoint scalar logits are non-finite")
        return logits, receipt

    support_contract = _support_owner_contract(context)
    endpoint_required = support_contract.get("status") == "measured"
    prefix = getattr(context, "prefix_ids", None)
    if not isinstance(prefix, torch.Tensor) or prefix.ndim != 2 or prefix.shape[0] != 1 or prefix.shape[1] < 2:
        if endpoint_required:
            raise OrchestrationError(
                "required active endpoint candidate scoring lacks a non-empty exact prefix boundary"
            )
        return _not_measured("active endpoint candidate scoring requires a non-empty exact prefix boundary")

    candidate_rows: dict[str, list[int]] = {}
    missing: dict[str, str] = {}
    target_row = getattr(context, "target_row_ids", ())
    if isinstance(target_row, Sequence) and not isinstance(target_row, (str, bytes)) and target_row:
        candidate_rows["target-B"] = [int(value) for value in target_row]
    else:
        missing["target-B"] = "target-B row tokens are absent from the event context"
        if endpoint_required:
            raise OrchestrationError(
                "required active endpoint candidate scoring lacks target-B row tokens"
            )

    covered_row = getattr(context, "latest_row_ids", ())
    if isinstance(covered_row, Sequence) and not isinstance(covered_row, (str, bytes)) and covered_row:
        candidate_rows["covered-A"] = [int(value) for value in covered_row]
    else:
        missing["covered-A"] = "covered-A native row tokens are absent from the event context"

    if support_contract.get("status") != "measured":
        missing["verified-uncovered"] = str(
            support_contract.get("reason", "independently verified support is not measured")
        )
    else:
        support_ids = [str(value) for value in support_contract.get("owner_ids", ())]
        covered_ids = {str(value) for value in getattr(context, "covered_owner_ids", ())}
        event = getattr(context, "event", {})
        target_owner = str(event.get("gt_owner_id")) if isinstance(event, Mapping) else ""
        for owner_id in sorted(set(support_ids) - covered_ids - {target_owner}):
            runtime = getattr(context, "runtime", None)
            row: Sequence[int] | None = None
            if runtime is not None:
                try:
                    row = _row_from_owner(runtime, owner_id, getattr(adapter, "tokenizer", None), contract)
                except (OrchestrationError, TypeError, ValueError):
                    row = None
            if row:
                candidate_rows[f"verified-uncovered:{owner_id}"] = [int(value) for value in row]
            else:
                missing[f"verified-uncovered:{owner_id}"] = (
                    "verified-uncovered owner has no source/derived row token identity"
                )
        if not (set(support_ids) - covered_ids - {target_owner}):
            missing["verified-uncovered"] = "support ledger is measured but has no uncovered owner beyond target/covered history"

    if not candidate_rows:
        if endpoint_required:
            raise OrchestrationError(
                "required active endpoint candidate scoring has no target-B candidate"
            )
        return _not_measured(
            "active endpoint candidate rows are unavailable",
            missing_candidates=missing,
            support_contract=support_contract,
        )

    try:
        boundary_logits, boundary_receipt = active_logits(prefix[:, :-1])
        opener_id = int(contract.object_ref_start_token_id)
        eos_id = contract.eos_token_id
        boundary_log_probs = torch.log_softmax(boundary_logits, dim=-1)
        def selected_margin(log_probs: torch.Tensor, token_id: int) -> tuple[float, int, float]:
            if token_id < 0 or token_id >= int(log_probs.shape[-1]) or int(log_probs.shape[-1]) < 2:
                raise OrchestrationError("active endpoint scalar forward has no competing token")
            competitors = log_probs.detach().clone()
            competitors[token_id] = -torch.inf
            competitor_id = int(torch.argmax(competitors).item())
            competitor_log_probability = float(competitors[competitor_id].item())
            selected_log_probability = float(log_probs[token_id].item())
            return (
                selected_log_probability - competitor_log_probability,
                competitor_id,
                competitor_log_probability,
            )
        boundary_margin, boundary_competitor_id, boundary_competitor_logprob = selected_margin(
            boundary_log_probs, opener_id
        )
        if eos_id is None:
            boundary_receipt_out: dict[str, Any] = _not_measured(
                "native wrapper contract has no EOS/<|im_end|> token"
            )
        else:
            values = torch.stack((boundary_log_probs[opener_id], boundary_log_probs[eos_id]))
            if not bool(torch.isfinite(values).all().item()):
                boundary_receipt_out = _not_measured(
                    "active pre-opener row-entry/STOP logits are non-finite"
                )
            else:
                boundary_receipt_out = {
                    "status": "measured",
                    "source": "active_intervention_pre_opener_scalar_forward",
                    "teacher_forced": True,
                    "diagnostic_only": True,
                    "active_intervention": str(active_intervention),
                    "row_entry_token_id": opener_id,
                    "terminal_token_id": int(eos_id),
                    "row_entry_log_probability": float(values[0].item()),
                    "terminal_log_probability": float(values[1].item()),
                    "row_entry_minus_terminal": float((values[0] - values[1]).item()),
                    "selected_vs_best_competing_token_margin": boundary_margin,
                    "best_competing_token_id": boundary_competitor_id,
                    "best_competing_token_log_probability": boundary_competitor_logprob,
                    "input_prefix_sha256": sha256_token_ids(prefix[:, :-1]),
                }
        forward_count = 1
        auxiliary_forward_count = int(boundary_receipt.get("auxiliary_forward_count", 0))
        scored: dict[str, Any] = {}
        for label, row in candidate_rows.items():
            phases = _segment_token_indices(row, contract=contract)
            selected: list[float] = [float(boundary_log_probs[opener_id].item())]
            margins: list[float] = [boundary_margin]
            competitor_ids: list[int] = [boundary_competitor_id]
            competitor_logprobs: list[float] = [boundary_competitor_logprob]
            current = prefix.detach().clone()
            row_receipts: list[Mapping[str, Any]] = [boundary_receipt]
            for token in row[1:]:
                logits, scalar_receipt = active_logits(current)
                log_probs = torch.log_softmax(logits, dim=-1)
                if token < 0 or token >= int(logits.shape[-1]):
                    raise OrchestrationError(f"{label} row token {token} is outside active logits vocabulary")
                selected_value = log_probs[token]
                selected.append(float(selected_value.item()))
                margin, competitor_id, competitor_logprob = selected_margin(log_probs, token)
                margins.append(margin)
                competitor_ids.append(competitor_id)
                competitor_logprobs.append(competitor_logprob)
                row_receipts.append(scalar_receipt)
                forward_count += 1
                auxiliary_forward_count += int(scalar_receipt.get("auxiliary_forward_count", 0))
                current = torch.cat(
                    (current, torch.tensor([[int(token)]], dtype=torch.long, device=current.device)),
                    dim=1,
                )
            if len(selected) != len(row):
                raise OrchestrationError(f"{label} active candidate score arity mismatch")
            selected_tensor = torch.tensor(selected, dtype=torch.float64)
            margin_tensor = torch.tensor(margins, dtype=torch.float64)
            segments: dict[str, Any] = {}
            for phase, indices in phases.items():
                values = selected_tensor[list(indices)]
                phase_margins = margin_tensor[list(indices)]
                segments[phase] = {
                    "status": "measured",
                    "token_indices": list(indices),
                    "token_count": len(indices),
                    "sum_log_probability": float(values.sum().item()),
                    "mean_log_probability": float(values.mean().item()),
                    "selected_vs_best_token_margins": [float(value) for value in phase_margins.tolist()],
                    "mean_selected_vs_best_token_margin": float(phase_margins.mean().item()),
                    "best_competing_token_ids": [competitor_ids[index] for index in indices],
                    "best_competing_token_log_probabilities": [competitor_logprobs[index] for index in indices],
                }
            scored[label] = {
                "status": "measured",
                "source": "active_intervention_scalar_forward",
                "teacher_forced": True,
                "diagnostic_only": True,
                "natural_release_claim": False,
                "active_intervention": str(active_intervention),
                "row_token_count": len(row),
                "row_token_ids_sha256": sha256_token_ids(row),
                "segments": segments,
                "selected_token_log_probabilities": selected,
                "selected_vs_best_token_margins": margins,
                "best_competing_token_ids": competitor_ids,
                "best_competing_token_log_probabilities": competitor_logprobs,
                "forward_receipt_count": len(row_receipts),
            }
        target = scored.get("target-B")
        comparisons: dict[str, Any] = {}
        if not isinstance(target, Mapping) or target.get("status") != "measured":
            comparisons["target_deltas"] = _not_measured("target-B active candidate score is unavailable")
        else:
            target_segments = target.get("segments", {})
            target_deltas: dict[str, Any] = {}
            for label, candidate in scored.items():
                candidate_segments = candidate.get("segments", {}) if isinstance(candidate, Mapping) else {}
                delta: dict[str, Any] = {}
                for phase in ("description", "geometry", "full_row"):
                    target_phase = target_segments.get(phase) if isinstance(target_segments, Mapping) else None
                    candidate_phase = candidate_segments.get(phase) if isinstance(candidate_segments, Mapping) else None
                    if isinstance(target_phase, Mapping) and isinstance(candidate_phase, Mapping):
                        delta[phase] = float(
                            candidate_phase["mean_log_probability"] - target_phase["mean_log_probability"]
                        )
                    else:
                        delta[phase] = _not_measured(f"{phase} score missing for {label}")
                target_deltas[label] = {
                    "status": "measured",
                    "candidate_minus_target_mean_log_probability": delta,
                    "target_minus_candidate_mean_log_probability": {
                        phase: (
                            -value
                            if isinstance(value, (int, float))
                            else value
                        )
                        for phase, value in delta.items()
                    },
                }
            comparisons["target_deltas"] = target_deltas
        return {
            "status": "measured",
            "source": "active_intervention_scalar_forward",
            "teacher_forced": True,
            "diagnostic_only": True,
            "natural_release_claim": False,
            "active_intervention": str(active_intervention),
            "candidates": scored,
            "missing_candidates": missing,
            "comparisons": comparisons,
            "row_entry_vs_native_im_end": boundary_receipt_out,
            "forward_count": forward_count,
            "auxiliary_forward_count": auxiliary_forward_count,
            "cost": {
                "mode": "scalar_exact",
                "active_forward_count": forward_count,
                "auxiliary_capture_forward_count": auxiliary_forward_count,
                "candidate_count": len(candidate_rows),
            },
            "support_contract": support_contract,
        }
    except (AttributeError, OrchestrationError, RuntimeError, TypeError, ValueError) as exc:
        if endpoint_required:
            raise OrchestrationError(
                f"required active endpoint candidate scalar scorer failed: {exc}"
            ) from exc
        return _not_measured(
            f"active endpoint candidate scalar scorer failed: {exc}",
            active_intervention=str(active_intervention),
            missing_candidates=missing,
            support_contract=support_contract,
        )


def helper_schema_receipt() -> dict[str, Any]:
    """Hash the exact experiment-local helper schemas used by this runner."""

    versions = {
        "static": static.SCHEMA_VERSION,
        "dynamic": "dynamic_history_and_crossover_probe.v1",
        "gradient": gradient.SCHEMA_VERSION,
    }
    return {
        "schema_version": HELPER_SCHEMA_VERSION,
        "versions": versions,
        "sha256": sha256_json(versions),
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestrationError(f"cannot read JSON artifact {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for value in values:
            handle.write(_canonical(dict(value)) + b"\n")


def _teacher_forced_row_offsets(
    *, prefix_length: int, row_lengths: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    """Return next-token-logit slices for rows appended to one native prefix.

    A causal LM logit at input position ``i`` predicts token ``i + 1``.  The
    first token of an appended row therefore uses the logit immediately before
    that row, while the final row token uses the logit immediately before the
    row's final input token.  Keeping this indexing in one checked helper
    prevents silently dropping the opener or scoring the following row token.
    """

    if isinstance(prefix_length, bool) or int(prefix_length) <= 0:
        raise OrchestrationError("teacher-forced row offsets require a non-empty prefix")
    offsets: list[tuple[int, int]] = []
    start = int(prefix_length)
    for index, raw_length in enumerate(row_lengths):
        if isinstance(raw_length, bool) or int(raw_length) <= 0:
            raise OrchestrationError(f"teacher-forced row {index} must be non-empty")
        length = int(raw_length)
        # [start - 1, start + length - 1) is exactly ``length`` next-token
        # logits aligned to target token positions [start, start + length).
        offsets.append((start - 1, start + length - 1))
        start += length
    return tuple(offsets)


def _native_arithmetic_position_view(
    hidden: torch.Tensor, positions: Sequence[int], label: str
) -> torch.Tensor:
    """Return a graph-preserving arithmetic span from contiguous [1,S,H] output."""

    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
        raise OrchestrationError(f"P4 {label} hidden output must have shape [1,S,H]")
    sequence_length = int(hidden.shape[1])
    width = int(hidden.shape[2])
    if not hidden.is_contiguous() or tuple(hidden.stride()) != (sequence_length * width, width, 1):
        raise OrchestrationError(
            f"P4 {label} hidden output must be contiguous [1,S,H] for a graph-preserving view"
        )
    selected = tuple(int(position) for position in positions)
    if not selected:
        raise OrchestrationError(f"P4 {label} position span is empty")
    if any(position < 0 or position >= sequence_length for position in selected):
        raise OrchestrationError(f"P4 {label} position exceeds diagnostic forward")
    if len(selected) == 1:
        step = 1
    else:
        differences = {right - left for left, right in zip(selected, selected[1:])}
        if len(differences) != 1 or next(iter(differences)) <= 0:
            raise OrchestrationError(
                f"P4 {label} positions are non-arithmetic; refusing a detached index gather"
            )
        step = next(iter(differences))
    return hidden[0].as_strided(
        (len(selected), width),
        (step * width, 1),
        storage_offset=selected[0] * width,
    )


def _build_token_mass_contract(
    adapter: ExperimentRuntimeAdapter,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    """Build the native grammar/STOP/invalid token-mass sets for P4."""

    tokenizer = getattr(adapter, "tokenizer", None)
    if tokenizer is None:
        raise OrchestrationError("P4 token-mass contract requires the native tokenizer")
    all_special_ids = getattr(tokenizer, "all_special_ids", None)
    if all_special_ids is None or isinstance(all_special_ids, (str, bytes)):
        raise OrchestrationError("P4 token-mass contract requires tokenizer.all_special_ids")
    try:
        special_values = list(all_special_ids)
    except TypeError as exc:
        raise OrchestrationError("tokenizer.all_special_ids must be an integer sequence") from exc
    vocab_size: int | None = None
    vocab_source = ""
    try:
        candidate = int(len(tokenizer))
    except (TypeError, ValueError):
        candidate = 0
    if candidate > 0:
        vocab_size = candidate
        vocab_source = "len(tokenizer)"
    else:
        raw_vocab_size = getattr(tokenizer, "vocab_size", None)
        if isinstance(raw_vocab_size, bool) or not isinstance(raw_vocab_size, int) or raw_vocab_size <= 0:
            raise OrchestrationError("P4 token-mass contract requires a positive tokenizer vocabulary size")
        vocab_size = int(raw_vocab_size)
        vocab_source = "tokenizer.vocab_size"

    wrapper = adapter.wrapper_contract
    coordinate_start = int(wrapper.coordinate_token_start_id)
    coordinate_count = int(wrapper.coordinate_bin_count)
    if coordinate_count <= 0:
        raise OrchestrationError("P4 wrapper coordinate_bin_count must be positive")

    def normalize(values: Iterable[int], label: str) -> tuple[int, ...]:
        result: set[int] = set()
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise OrchestrationError(f"P4 {label} token IDs must be integers")
            if value < 0 or value >= vocab_size:
                raise OrchestrationError(f"P4 {label} token ID {value} is outside vocabulary size {vocab_size}")
            result.add(int(value))
        if not result:
            raise OrchestrationError(f"P4 {label} token set must be non-empty")
        return tuple(sorted(result))

    grammar_ids = normalize(
        (
            wrapper.object_ref_start_token_id,
            wrapper.object_ref_end_token_id,
            wrapper.box_start_token_id,
            wrapper.box_end_token_id,
            *range(coordinate_start, coordinate_start + coordinate_count),
            *((wrapper.commit_token_id,) if wrapper.commit_token_id is not None else ()),
        ),
        "grammar",
    )
    stop_ids = normalize(
        tuple(value for value in (wrapper.closure_token_id, wrapper.eos_token_id) if value is not None),
        "STOP",
    )
    special_in_vocab: list[int] = []
    for value in special_values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise OrchestrationError("tokenizer.all_special_ids must contain only integer IDs")
        if 0 <= value < vocab_size:
            special_in_vocab.append(int(value))
    invalid_ids = tuple(sorted(set(special_in_vocab) - set(grammar_ids) - set(stop_ids)))
    if not invalid_ids:
        raise OrchestrationError(
            "P4 token-mass invalid set is empty after excluding grammar/STOP special IDs"
        )

    def set_receipt(values: Sequence[int]) -> dict[str, Any]:
        return {
            "ids": list(values),
            "count": len(values),
            "sha256": sha256_json(list(values)),
        }

    receipt = {
        "schema_version": "owner_interface.token_mass_contract.v1",
        "vocab_size": vocab_size,
        "vocab_size_source": vocab_source,
        "grammar": set_receipt(grammar_ids),
        "stop": set_receipt(stop_ids),
        "invalid": set_receipt(invalid_ids),
        "semantics": {
            "grammar": "object-ref/box markers plus complete coordinate token range and commit when native",
            "stop": "native row closure plus tokenizer EOS when present",
            "invalid": "tokenizer special IDs in vocabulary excluding grammar and STOP",
        },
    }
    return grammar_ids, stop_ids, invalid_ids, receipt


def _load_h0_ledger_records(
    *,
    sources: Mapping[str, Any],
    manifest_sources: Mapping[str, Any],
    checkpoint: Literal["S", "A"],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, str]]:
    """Load checkpoint-native boundary records keyed by image and target owner."""

    raw_ledgers = sources.get("h0_ledgers")
    manifest_ledgers = manifest_sources.get("h0_ledgers")
    if not isinstance(raw_ledgers, list) or not raw_ledgers:
        raise OrchestrationError("cohort sources.h0_ledgers must be a non-empty list")
    if not isinstance(manifest_ledgers, list) or len(manifest_ledgers) != len(raw_ledgers):
        raise OrchestrationError("cohort manifest h0_ledgers hashes do not match source entries")
    by_image: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    boundary_provenance_by_image: dict[str, str] = {}
    ledger_identity: dict[str, str] = {}
    for index, (source, manifest_hash) in enumerate(zip(raw_ledgers, manifest_ledgers, strict=True)):
        if not isinstance(source, Mapping):
            raise OrchestrationError(f"cohort sources.h0_ledgers[{index}] must be a mapping")
        path_value = source.get("path")
        declared_hash = source.get("sha256")
        if not isinstance(path_value, str) or not isinstance(declared_hash, str):
            raise OrchestrationError(f"cohort sources.h0_ledgers[{index}] requires path and sha256")
        if declared_hash != manifest_hash or sha256_file(Path(path_value)) != declared_hash:
            raise OrchestrationError(f"checkpoint-native H0 ledger hash mismatch: {path_value}")
        payload = _read_json(Path(path_value))
        if payload.get("checkpoint") != checkpoint or payload.get("unit_id") != UNIT_ID:
            raise OrchestrationError(f"H0 ledger identity mismatch: {path_value}")
        records = payload.get("records")
        if not isinstance(records, list):
            raise OrchestrationError(f"H0 ledger records must be a list: {path_value}")
        for record_index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise OrchestrationError(f"H0 ledger record {record_index} is not a mapping")
            image_id = str(record.get("image_id"))
            owner_id = record.get("gt_owner_id")
            covered = record.get("covered_owner_ids")
            boundary = record.get("natural_boundary")
            if image_id in {"None", ""} or not isinstance(owner_id, str):
                raise OrchestrationError(f"H0 ledger record {record_index} lacks image/owner identity")
            if isinstance(boundary, bool) or not isinstance(boundary, int) or boundary < 0:
                raise OrchestrationError(f"H0 ledger record {image_id}/{owner_id} has invalid natural boundary")
            if not isinstance(covered, list) or any(not isinstance(value, str) for value in covered):
                raise OrchestrationError(f"H0 ledger record {image_id}/{owner_id} has invalid covered_owner_ids")
            if len(set(covered)) != len(covered):
                raise OrchestrationError(f"H0 ledger record {image_id}/{owner_id} repeats covered owners")
            exact_prefix = record.get("exact_prefix_token_ids")
            exact_prefix_sha256 = record.get("exact_prefix_sha256")
            if (
                not isinstance(exact_prefix, list)
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in exact_prefix)
                or not isinstance(exact_prefix_sha256, str)
                or sha256_token_ids(exact_prefix) != exact_prefix_sha256
            ):
                raise OrchestrationError(
                    f"H0 ledger record {image_id}/{owner_id} has an invalid exact generated-history prefix"
                )
            latest = record.get("latest_covered_owner_id")
            if covered and latest != covered[-1]:
                raise OrchestrationError(f"H0 ledger record {image_id}/{owner_id} latest owner is not boundary-last")
            if not covered and latest is not None:
                raise OrchestrationError(f"H0 ledger record {image_id}/{owner_id} has latest owner without coverage")
            valid_prediction_count = record.get("valid_prediction_count")
            boundaries = record.get("generated_row_boundaries")
            if (
                isinstance(valid_prediction_count, bool)
                or not isinstance(valid_prediction_count, int)
                or valid_prediction_count < 0
            ):
                raise OrchestrationError(
                    f"H0 ledger record {image_id}/{owner_id} has invalid valid_prediction_count"
                )
            if not isinstance(boundaries, list) or len(boundaries) != valid_prediction_count:
                raise OrchestrationError(
                    f"H0 ledger record {image_id}/{owner_id} generated-row boundaries "
                    "do not match valid_prediction_count"
                )
            expected_closure_id = 151669 if checkpoint == "A" else 151649
            expected_closure_text = "<|commit|>" if checkpoint == "A" else "<|box_end|>"
            normalized_boundaries: list[dict[str, Any]] = []
            seen_prediction_indices: set[int] = set()
            seen_tp_owners: set[str] = set()
            previous_generated_order = -1
            previous_closure_step = -1
            for boundary_index, raw_boundary in enumerate(boundaries):
                if not isinstance(raw_boundary, Mapping):
                    raise OrchestrationError(
                        f"H0 ledger record {image_id}/{owner_id} generated-row boundary "
                        f"{boundary_index} is not a mapping"
                    )

                def boundary_int(key: str) -> int:
                    value = raw_boundary.get(key)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise OrchestrationError(
                            f"H0 ledger record {image_id}/{owner_id} generated-row boundary "
                            f"{boundary_index} has invalid {key}"
                        )
                    return int(value)

                prediction_index = boundary_int("prediction_index")
                generated_order = boundary_int("generated_order")
                row_start_step = boundary_int("row_start_step")
                span_end_step = boundary_int("span_end_step")
                closure_step = boundary_int("closure_step")
                if prediction_index in seen_prediction_indices:
                    raise OrchestrationError(
                        f"H0 ledger record {image_id}/{owner_id} repeats a prediction index"
                    )
                if generated_order <= previous_generated_order:
                    raise OrchestrationError(
                        f"H0 ledger record {image_id}/{owner_id} generated-row boundaries "
                        "are not ordered by generated_order"
                    )
                if not (
                    previous_closure_step < row_start_step <= span_end_step <= closure_step
                ):
                    raise OrchestrationError(
                        f"H0 ledger record {image_id}/{owner_id} has incoherent or overlapping "
                        "generated-row steps"
                    )
                closure_token_id = raw_boundary.get("closure_token_id")
                closure_token_text = raw_boundary.get("closure_token_text")
                if (
                    isinstance(closure_token_id, bool)
                    or closure_token_id != expected_closure_id
                    or closure_token_text != expected_closure_text
                ):
                    raise OrchestrationError(
                        f"H0 ledger record {image_id}/{owner_id} generated-row boundary "
                        f"{boundary_index} lacks the checkpoint-native closure token"
                    )
                commit_step = raw_boundary.get("commit_step")
                if checkpoint == "A":
                    if (
                        isinstance(commit_step, bool)
                        or not isinstance(commit_step, int)
                        or commit_step < 0
                        or commit_step != closure_step
                        or closure_step != span_end_step + 1
                    ):
                        raise OrchestrationError(
                            f"H0 ledger record {image_id}/{owner_id} generated-row boundary "
                            f"{boundary_index} has an invalid native commit step"
                        )
                elif commit_step is not None or closure_step != span_end_step:
                    raise OrchestrationError(
                        f"H0 ledger record {image_id}/{owner_id} generated-row boundary "
                        f"{boundary_index} has an invalid closed-wrapper closure step"
                    )
                match_status = raw_boundary.get("match_status")
                matched_owner_id = raw_boundary.get("gt_owner_id")
                if match_status == "tp":
                    if (
                        not isinstance(matched_owner_id, str)
                        or not matched_owner_id
                        or matched_owner_id in seen_tp_owners
                    ):
                        raise OrchestrationError(
                            f"H0 ledger record {image_id}/{owner_id} generated-row boundary "
                            f"{boundary_index} has an invalid globally matched owner"
                        )
                    seen_tp_owners.add(matched_owner_id)
                elif match_status == "unmatched":
                    if matched_owner_id is not None:
                        raise OrchestrationError(
                            f"H0 ledger record {image_id}/{owner_id} unmatched generated-row "
                            f"boundary {boundary_index} names an owner"
                        )
                else:
                    raise OrchestrationError(
                        f"H0 ledger record {image_id}/{owner_id} generated-row boundary "
                        f"{boundary_index} has invalid match_status"
                    )
                normalized_boundaries.append(dict(raw_boundary))
                seen_prediction_indices.add(prediction_index)
                previous_generated_order = generated_order
                previous_closure_step = closure_step
            if seen_prediction_indices != set(range(valid_prediction_count)):
                raise OrchestrationError(
                    f"H0 ledger record {image_id}/{owner_id} generated-row boundaries "
                    "are incomplete for valid_prediction_count"
                )
            boundary_provenance = sha256_json({
                "valid_prediction_count": valid_prediction_count,
                "generated_row_boundaries": normalized_boundaries,
            })
            observed_provenance = boundary_provenance_by_image.setdefault(
                image_id, boundary_provenance
            )
            if observed_provenance != boundary_provenance:
                raise OrchestrationError(
                    f"H0 ledger records for image {image_id} disagree on generated-row provenance"
                )
            if owner_id in by_image[image_id]:
                raise OrchestrationError(f"H0 ledger repeats image/owner record {image_id}/{owner_id}")
            normalized_record = {
                "natural_boundary": boundary,
                "covered_owner_ids": tuple(covered),
                "latest_covered_owner_id": latest,
                "exact_prefix_token_ids": tuple(exact_prefix),
                "exact_prefix_sha256": exact_prefix_sha256,
                "strict_complete_row": record.get("strict_complete_row"),
                "valid_prediction_count": valid_prediction_count,
                "generated_row_boundaries": normalized_boundaries,
                # Support ledgers are merged into the H0 record by the cohort
                # materializer.  Preserve the claim/reason additively so
                # endpoint/P4 receipts can distinguish missing support from a
                # measured empty set without inventing a zero.
                "verified_support": record.get("verified_support"),
                "support_status": record.get("support_status"),
                "support_reason": record.get("support_reason"),
                "native_tp": record.get("native_tp"),
                "native_fn": record.get("native_fn"),
                "source_path": str(Path(path_value).expanduser().resolve()),
                "record_index": record_index,
            }
            due_boundary_evidence = record.get("due_boundary_evidence")
            if isinstance(due_boundary_evidence, Mapping):
                normalized_record["due_boundary_evidence"] = dict(due_boundary_evidence)
            by_image[image_id][owner_id] = normalized_record
        ledger_identity[str(Path(path_value).expanduser().resolve())] = declared_hash
    return {image: dict(records) for image, records in by_image.items()}, ledger_identity


def _resolve_h0_artifact(
    checkpoint: Literal["S", "A"],
    *,
    h0_root: Path,
    explicit: Path | None = None,
) -> Path:
    """Select a completed H0 root without assuming a repair suffix."""

    candidates = [explicit.expanduser().resolve()] if explicit is not None else sorted(
        path for path in h0_root.expanduser().resolve().glob(f"*{CHECKPOINTS[checkpoint]['h0_name']}*") if path.is_dir()
    )
    valid: list[Path] = []
    for root in candidates:
        summary_path = root / "summary.json"
        manifest_path = root / "run_manifest.json"
        if not summary_path.is_file() or not manifest_path.is_file():
            continue
        summary = _read_json(summary_path)
        manifest = _read_json(manifest_path)
        if summary.get("terminal_status") != "completed" or manifest.get("terminal_status") != "completed":
            continue
        if manifest.get("backend") != "hf" or manifest.get("backend_mode") != "generate":
            raise OrchestrationError(f"H0 artifact {root} is not an HF generate run")
        required = [root / name for name in ("pred_token_trace.jsonl", "image_plan.jsonl")]
        if not all(path.is_file() for path in required):
            continue
        valid.append(root)
    if not valid:
        raise OrchestrationError(
            f"no successful H0 artifact for checkpoint {checkpoint} under {h0_root}; "
            "a failed/partial H0 cannot seed a natural-prefix experiment"
        )
    # Explicit paths are exact.  Discovery chooses the lexicographically last
    # completed repair, while preserving the path in the identity receipt.
    return valid[-1]


def _trace_rows(path: Path, *, contract: static.WrapperContract) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OrchestrationError(f"invalid H0 trace JSON at {path}:{line_no}") from exc
            # Selected-token replay rows are not natural generation.  Keep only
            # the backend's literal generated token stream and remove padding.
            if item.get("trace_type") != "generated_token" or item.get("is_pad"):
                continue
            if "token_id" not in item or "row_id" not in item:
                continue
            grouped[str(item["row_id"])].append(item)
    result: dict[str, dict[str, Any]] = {}
    for row_id, items in grouped.items():
        items.sort(key=lambda item: int(item.get("generated_step_index", 0)))
        tokens: list[int] = []
        stop_reason = "length"
        for item in items:
            token = int(item["token_id"])
            if token == contract.eos_token_id or bool(item.get("is_stop")):
                stop_reason = "im_end"
                break
            tokens.append(token)
        if not tokens:
            raise OrchestrationError(f"H0 trace {row_id} has no natural generated tokens")
        starts = [index for index, token in enumerate(tokens) if token == contract.object_ref_start_token_id]
        if not starts or starts[0] != 0:
            raise OrchestrationError(f"H0 trace {row_id} does not begin with object_ref_start")
        rows: list[list[int]] = []
        for start, end in zip(starts, [*starts[1:], len(tokens)], strict=True):
            row = tokens[start:end]
            if len(row) > MAX_NATIVE_ROW_TOKENS:
                raise OrchestrationError(f"H0 row {row_id}:{len(rows)} exceeds native row budget")
            parsed = contract.parse_generated_suffix(row[1:])
            if not parsed.get("valid"):
                raise OrchestrationError(
                    f"H0 row {row_id}:{len(rows)} failed native {contract.assistant_format} parser: {parsed.get('reason')}"
                )
            rows.append(row)
        result[row_id] = {
            "row_id": row_id,
            "generated_token_ids": tokens,
            "generated_token_ids_sha256": sha256_token_ids(tokens),
            "rows": rows,
            "row_token_ids_sha256": [sha256_token_ids(row) for row in rows],
            "stop_reason": stop_reason,
            "trace_sha256": sha256_file(path),
        }
    if not result:
        raise OrchestrationError(f"H0 trace has no generated rows: {path}")
    return result


def _index_trace_rows_by_image(
    trace_rows: Mapping[str, dict[str, Any]],
    image_ids: Collection[str],
) -> dict[str, dict[str, Any]]:
    """Bind canonical H0 row IDs to numeric panel image IDs without guessing."""

    known = {str(image_id) for image_id in image_ids}
    indexed: dict[str, dict[str, Any]] = {}
    for row_id, row in trace_rows.items():
        key: str | None = row_id if row_id in known else None
        if key is None:
            match = re.search(r"(\d+)$", str(row_id))
            if match is not None:
                candidate = str(int(match.group(1)))
                if candidate in known:
                    key = candidate
        if key is None:
            continue
        if key in indexed:
            raise OrchestrationError(f"H0 trace rows collide on panel image identity {key}")
        indexed[key] = row
    return indexed


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1, iy1, ix2, iy2 = max(lx1, rx1), max(ly1, ry1), min(lx2, rx2), min(ly2, ry2)
    intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    left_area = max(lx2 - lx1, 0.0) * max(ly2 - ly1, 0.0)
    right_area = max(rx2 - rx1, 0.0) * max(ry2 - ry1, 0.0)
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _raw_objects(raw: Any) -> list[Any]:
    values = getattr(raw, "objects", None)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        return list(values)
    if (
        isinstance(raw, Mapping)
        and isinstance(raw.get("objects"), Sequence)
        and not isinstance(raw.get("objects"), (str, bytes, bytearray))
    ):
        return list(raw["objects"])
    return []


def _raw_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _raw_image_id(raw: Any) -> str:
    value = _raw_field(raw, "image_id")
    if value is None:
        metadata = _raw_field(raw, "metadata", {})
        source = metadata.get("source") if isinstance(metadata, Mapping) else None
        if isinstance(source, Mapping):
            value = source.get("image_id")
    if value is None:
        value = _raw_field(raw, "example_id", "unknown")
    return str(value)


def _select_event_panel_rows(raw_values: Sequence[Any], events: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Restrict runtime materialization to images admitted by this event shard."""

    by_image: dict[str, Any] = {}
    for raw in raw_values:
        image_id = _raw_image_id(raw)
        if image_id in by_image:
            raise OrchestrationError(f"derived panel contains duplicate image identities: {image_id}")
        by_image[image_id] = raw
    event_image_ids: set[str] = set()
    for index, event in enumerate(events):
        image_id = event.get("image_id")
        if image_id is None:
            raise OrchestrationError(f"event {index} lacks image_id for runtime materialization")
        event_image_ids.add(str(image_id))
    missing = sorted(event_image_ids - set(by_image))
    if missing:
        raise OrchestrationError(f"event images are absent from the derived panel: {missing}")
    return [raw for raw in raw_values if _raw_image_id(raw) in event_image_ids]


def _raw_dimensions(raw: Any) -> tuple[int, int]:
    image = raw.get("image", {}) if isinstance(raw, Mapping) else getattr(raw, "image", None)
    width = _raw_field(raw, "width", _raw_field(image, "width", 1000))
    height = _raw_field(raw, "height", _raw_field(image, "height", 1000))
    try:
        width_value, height_value = int(width), int(height)
    except (TypeError, ValueError) as exc:
        raise OrchestrationError("panel image width/height must be integers") from exc
    if width_value <= 0 or height_value <= 0:
        raise OrchestrationError("panel image width/height must be positive")
    return width_value, height_value


def _raw_coordinate_bins(raw: Any, obj: Any) -> tuple[int, int, int, int] | None:
    bbox = _raw_field(obj, "bbox", _raw_field(obj, "bbox_2d"))
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes, bytearray)) or len(bbox) != 4:
        return None
    bins: list[int] = []
    for value in bbox:
        if isinstance(value, str):
            match = re.fullmatch(r"<\|coord_(\d+)\|>", value.strip())
            if match is None:
                match = re.fullmatch(r"coord_(\d+)", value.strip())
            if match is None:
                return None
            parsed = int(match.group(1))
        else:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
        if parsed < 0 or parsed > 999:
            return None
        bins.append(parsed)
    return tuple(bins)  # type: ignore[return-value]


def _raw_object_descriptor(raw: Any, obj: Any, index: int) -> dict[str, Any]:
    width, height = _raw_dimensions(raw)
    bins = _raw_coordinate_bins(raw, obj)
    if bins is None:
        raise OrchestrationError(f"panel image {_raw_image_id(raw)} object {index} lacks a valid four-coordinate box")
    pixel_bbox = tuple(
        int(round(value * (width if axis % 2 == 0 else height) / 1000.0))
        for axis, value in enumerate(bins)
    )
    category = str(
        _raw_field(
            obj,
            "description",
            _raw_field(obj, "desc", _raw_field(obj, "category_name", _raw_field(obj, "category", ""))),
        )
    ).strip().lower()
    if not category:
        raise OrchestrationError(f"panel image {_raw_image_id(raw)} object {index} lacks a category")
    # COCO annotation identity is intentionally read from the object top level.
    # ``load_raw_examples`` materializes the original top-level field as its
    # ``object_id`` field; nested metadata is never used as a fallback identity.
    coco_ann_id = _raw_field(obj, "coco_ann_id")
    if coco_ann_id is None:
        coco_ann_id = _raw_field(obj, "object_id")
    if isinstance(coco_ann_id, bool):
        raise OrchestrationError(f"panel image {_raw_image_id(raw)} object {index} has an invalid coco_ann_id")
    return {
        "source_index": int(index),
        "category": category,
        "pixel_bbox": list(pixel_bbox),
        "bbox_bins": list(bins),
        "coco_ann_id": coco_ann_id,
    }


def _raw_owner_rows(raw: Any) -> list[dict[str, Any]]:
    image_id = _raw_image_id(raw)
    owners: list[dict[str, Any]] = []
    for index, obj in enumerate(_raw_objects(raw)):
        descriptor = _raw_object_descriptor(raw, obj, index)
        owners.append({
            "owner_id": f"gt:{image_id}:{index}",
            "source_index": index,
            "category": descriptor["category"],
            "pixel_bbox": descriptor["pixel_bbox"],
            "coco_ann_id": descriptor["coco_ann_id"],
        })
    return owners


def _build_source_derived_owner_mapping(
    source_raw: Any,
    derived_raw: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Bind source owner identities to the reordered derived panel exactly once."""

    source_image_id = _raw_image_id(source_raw)
    derived_image_id = _raw_image_id(derived_raw)
    if source_image_id != derived_image_id:
        raise OrchestrationError(
            f"source/derived panel image identity mismatch: {source_image_id} != {derived_image_id}"
        )
    source_objects = _raw_objects(source_raw)
    derived_objects = _raw_objects(derived_raw)
    if len(source_objects) != len(derived_objects):
        raise OrchestrationError(
            f"image {source_image_id} source/derived owner count mismatch: "
            f"{len(source_objects)} != {len(derived_objects)}"
        )
    source_descriptors = [_raw_object_descriptor(source_raw, obj, index) for index, obj in enumerate(source_objects)]
    derived_descriptors = [_raw_object_descriptor(derived_raw, obj, index) for index, obj in enumerate(derived_objects)]
    used_derived: set[int] = set()
    owner_mapping: dict[str, dict[str, Any]] = {}
    receipt_rows: list[dict[str, Any]] = []
    for source in source_descriptors:
        source_index = int(source["source_index"])
        ann_id = source.get("coco_ann_id")
        ann_matches = [
            derived
            for derived in derived_descriptors
            if ann_id is not None
            and derived.get("coco_ann_id") is not None
            and str(derived["coco_ann_id"]) == str(ann_id)
        ]
        method: str
        if len(ann_matches) > 1:
            raise OrchestrationError(
                f"image {source_image_id} source owner {source_index} has ambiguous coco_ann_id {ann_id!r}"
            )
        if len(ann_matches) == 1:
            derived = ann_matches[0]
            method = "coco_ann_id"
            if int(derived["source_index"]) in used_derived:
                raise OrchestrationError(
                    f"image {source_image_id} coco_ann_id mapping is not one-to-one for {ann_id!r}"
                )
        else:
            candidates = [
                derived
                for derived in derived_descriptors
                if int(derived["source_index"]) not in used_derived
                and derived["category"] == source["category"]
                and tuple(derived["pixel_bbox"]) == tuple(source["pixel_bbox"])
            ]
            if len(candidates) != 1:
                reason = "ambiguous" if len(candidates) > 1 else "missing"
                raise OrchestrationError(
                    f"image {source_image_id} source owner {source_index} has {reason} "
                    "category/pixel-bbox derived identity"
                )
            derived = candidates[0]
            method = "category_pixel_bbox"
        derived_index = int(derived["source_index"])
        used_derived.add(derived_index)
        owner_id = f"gt:{source_image_id}:{source_index}"
        mapping = {
            "owner_id": owner_id,
            "source_index": source_index,
            "derived_index": derived_index,
            "coco_ann_id": ann_id,
            "mapping_method": method,
            "category": source["category"],
            "source_pixel_bbox": list(source["pixel_bbox"]),
            "derived_pixel_bbox": list(derived["pixel_bbox"]),
        }
        owner_mapping[owner_id] = mapping
        receipt_rows.append(dict(mapping))
    if used_derived != set(range(len(derived_descriptors))):
        raise OrchestrationError(f"image {source_image_id} source/derived owner mapping is not bijective")
    owners = _raw_owner_rows(source_raw)
    for owner in owners:
        mapping = owner_mapping.get(str(owner["owner_id"]))
        if mapping is None:
            raise OrchestrationError(f"image {source_image_id} source owner mapping is incomplete")
        owner.update(
            {
                "derived_index": mapping["derived_index"],
                "mapping_method": mapping["mapping_method"],
            }
        )
    receipt = {
        "schema_version": "owner_interface.source_derived_mapping.v1",
        "image_id": source_image_id,
        "source_owner_count": len(source_descriptors),
        "derived_owner_count": len(derived_descriptors),
        "source_to_derived": receipt_rows,
        "mapping_sha256": sha256_json(receipt_rows),
        "mapping_method_census": {
            method: sum(1 for row in receipt_rows if row["mapping_method"] == method)
            for method in sorted({str(row["mapping_method"]) for row in receipt_rows})
        },
    }
    return owners, owner_mapping, receipt


def source_specific_owner_match(
    parsed_prediction: Mapping[str, Any],
    owners: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Match one native parsed row to a unique source owner, never by index."""

    category = str(parsed_prediction.get("description", "")).strip().lower()
    bbox = parsed_prediction.get("bbox")
    candidates = [
        {
            **dict(owner),
            "iou": _bbox_iou(bbox or (), owner.get("pixel_bbox", ())),
        }
        for owner in owners
        if str(owner.get("category", "")).strip().lower() == category
    ]
    candidates = [item for item in candidates if float(item["iou"]) >= float(iou_threshold)]
    candidates.sort(key=lambda item: (-float(item["iou"]), str(item.get("owner_id"))))
    if not candidates:
        return {"status": "unmatched", "owner_id": None, "candidates": []}
    if len(candidates) > 1 and float(candidates[0]["iou"]) == float(candidates[1]["iou"]):
        return {"status": "ambiguous", "owner_id": None, "candidates": candidates}
    return {
        "status": "unique",
        "owner_id": str(candidates[0]["owner_id"]),
        "iou": float(candidates[0]["iou"]),
        "candidates": candidates,
    }


def _native_parse_row(
    row_tokens: Sequence[int],
    *,
    contract: static.WrapperContract,
    tokenizer: Any,
    row_id: str,
    row_index: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Run both the experiment token parser and the checkpoint-native parser."""

    parsed = contract.parse_generated_suffix(row_tokens[1:], tokenizer=tokenizer)
    if not parsed.get("valid"):
        return {"valid": False, "parse_status": "malformed", "reason": parsed.get("reason"), "row_token_ids": list(row_tokens)}
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise OrchestrationError("native owner matching requires tokenizer.decode")
    text = str(decode(list(row_tokens), skip_special_tokens=False))
    try:
        from src.inference.parsing import parse_compact_object_box

        native = parse_compact_object_box(
            text,
            assistant_format=("object_box_commit" if contract.commit_token_id is not None else "object_box_closed"),
            row_id=row_id,
            row_index=row_index,
            image_width=width,
            image_height=height,
        )
        native_dict = native.to_artifact_dict()
    except Exception as exc:  # parser contract failures are explicit below
        native_dict = {"parse_status": "parser_error", "error": str(exc), "predictions": []}
    predictions = native_dict.get("predictions", [])
    valid = bool(native_dict.get("parse_status") == "accepted" and len(predictions) == 1)
    return {
        "valid": valid,
        "parse_status": native_dict.get("parse_status"),
        "native": native_dict,
        "prediction": predictions[0] if valid else None,
        "row_token_ids": [int(value) for value in row_tokens],
        "row_token_ids_sha256": sha256_token_ids(row_tokens),
    }


@dataclass
class ImageRuntime:
    image_id: str
    raw: Any
    prompt_ids: torch.Tensor
    prompt_record: Any
    native_inputs: Mapping[str, Any]
    image_grid_thw: torch.Tensor
    image_token_id: int
    merge_size: int
    image_span: static.ImageSpan
    owners: list[dict[str, Any]]
    h0: dict[str, Any]
    width: int
    height: int
    owner_mapping: dict[str, dict[str, Any]] = field(default_factory=dict)
    mapping_receipt: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventContext:
    event: Mapping[str, Any]
    runtime: ImageRuntime
    prefix_ids: torch.Tensor
    natural_boundary: int
    latest_row_ids: list[int]
    covered_owner_ids: tuple[str, ...]
    target_row_ids: list[int]
    uncovered_row_ids: list[int]
    prefix_receipt: dict[str, Any] = field(default_factory=dict)
    actuator_eligible: bool = True
    eligibility_reason: str | None = None
    pair_status: str | None = None
    completed_row_ids: tuple[tuple[int, ...], ...] = ()
    checkpoint: str | None = None


@dataclass
class ExperimentRuntimeAdapter:
    """Live HF runtime plus seam methods consumed by the orchestrator."""

    config: Any
    resolved: Any
    frontend: Any
    session: Any
    model: Any
    tokenizer: Any
    processor: Any
    checkpoint: str
    h0_root: Path
    panel_path: Path
    panel_rows: dict[str, Any]
    h0_rows: dict[str, dict[str, Any]]
    h0_ledger_records: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    event_parser_calls: int = 0

    def close(self) -> None:
        if self.session is not None and callable(getattr(self.session, "close", None)):
            self.session.close()

    @property
    def model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    def exact_model_inputs(
        self,
        runtime: ImageRuntime,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], torch.Tensor, str]:
        device = self.model_device
        ids = input_ids.to(device=device, dtype=torch.long)
        ordinary_mask = torch.ones_like(ids, dtype=torch.long) if attention_mask is None else attention_mask.to(device=device)
        try:
            position_ids = static._position_ids(
                self.model,
                input_ids=ids,
                image_grid_thw=runtime.image_grid_thw,
                position_ids_builder=None,
            )
        except (TypeError, ValueError) as exc:
            raise OrchestrationError(f"runtime image_grid_thw/position_ids contract failed: {exc}") from exc
        payload = {
            key: value.to(device=device) if isinstance(value, torch.Tensor) else value
            for key, value in runtime.native_inputs.items()
            if key not in {"input_ids", "attention_mask", "position_ids", "cache_position", "rope_deltas", "token_type_ids"}
        }
        payload.update({"input_ids": ids, "attention_mask": ordinary_mask, "position_ids": position_ids, "use_cache": False, "return_dict": True, "logits_to_keep": 0})
        native_grid = payload.get("image_grid_thw", runtime.image_grid_thw)
        if not isinstance(native_grid, torch.Tensor):
            raise OrchestrationError("native payload image_grid_thw must be a tensor")
        if native_grid.ndim == 1 and native_grid.numel() == 3:
            normalized_native_grid = native_grid.reshape(1, 3)
        elif native_grid.ndim == 2 and int(native_grid.shape[1]) == 3:
            normalized_native_grid = native_grid
        else:
            raise OrchestrationError("native payload image_grid_thw must have shape [3] or [N,3]")
        stored_grid = runtime.image_grid_thw.to(device=device, dtype=torch.long).reshape(-1, 3)
        normalized_native_grid = normalized_native_grid.to(device=device, dtype=torch.long)
        if not torch.equal(stored_grid, normalized_native_grid):
            raise OrchestrationError("native payload image_grid_thw differs from the stored runtime grid")
        mrope_hash = dynamic.compute_mrope_hash(
            position_ids,
            image_grid_thw=native_grid,
            rope_deltas=payload.get("rope_deltas"),
        )
        return payload, position_ids, mrope_hash

    def forward(self, runtime: ImageRuntime, input_ids: torch.Tensor, *, attention_mask: torch.Tensor | None = None, context: Any = None) -> Any:
        payload, _position_ids, _mrope = self.exact_model_inputs(runtime, input_ids, attention_mask=attention_mask)
        with context if context is not None else nullcontext():
            with torch.inference_mode():
                return self.model(**payload)

    def parse_row(self, row_tokens: Sequence[int], runtime: ImageRuntime, *, row_index: int) -> dict[str, Any]:
        self.event_parser_calls += 1
        parsed = _native_parse_row(
            row_tokens,
            contract=self.wrapper_contract,
            tokenizer=self.tokenizer,
            row_id=runtime.image_id,
            row_index=row_index,
            width=runtime.width,
            height=runtime.height,
        )
        if parsed.get("prediction") is not None:
            owner_match = source_specific_owner_match(parsed["prediction"], runtime.owners)
            parsed["owner_match"] = {
                **owner_match,
                "source_specific": True,
                "physical_match": owner_match.get("status") == "unique",
            }
        else:
            parsed["owner_match"] = {"status": "unmatched", "owner_id": None, "candidates": []}
        return parsed

    @property
    def wrapper_contract(self) -> static.WrapperContract:
        cfg = self.config.template
        tokenizer = self.tokenizer
        def token(name: str) -> int:
            return int(tokenizer.convert_tokens_to_ids(name))
        commit = token("<|commit|>") if cfg.assistant_format == "object_box_commit" else None
        eos = getattr(tokenizer, "eos_token_id", None)
        return static.WrapperContract(
            assistant_format=cfg.assistant_format,
            object_ref_start_token_id=token("<|object_ref_start|>"),
            object_ref_end_token_id=token("<|object_ref_end|>"),
            box_start_token_id=token("<|box_start|>"),
            box_end_token_id=token("<|box_end|>"),
            coordinate_token_start_id=token("<|coord_0|>"),
            commit_token_id=commit,
            eos_token_id=None if eos is None else int(eos),
        )


@dataclass
class FakeRuntimeAdapter:
    """Small protocol adapter used by the integration tests only."""

    events: list[Mapping[str, Any]]
    test_only: bool = True
    parser_calls: int = 0
    helper_calls: list[str] = field(default_factory=list)
    persistent_calls: list[tuple[str, int]] = field(default_factory=list)

    def close(self) -> None:
        return None

    def run_event(self, event: Mapping[str, Any], *, checkpoint: str, stage: str) -> dict[str, Any]:
        """Exercise the same helper contracts without constructing a HF model."""

        del checkpoint, stage
        contract = static.WrapperContract(
            assistant_format="object_box_closed",
            object_ref_start_token_id=100,
            object_ref_end_token_id=101,
            box_start_token_id=102,
            box_end_token_id=103,
            coordinate_token_start_id=200,
        )
        parsed = contract.parse_generated_suffix([110, 101, 102, 200, 201, 202, 203, 103])
        if not parsed.get("valid"):
            raise OrchestrationError("fake parser contract failed")
        self.parser_calls += 1
        static.compare_noop_receipts(
            {"generated_token_ids": [1], "selected_token_log_probabilities": [0.0], "selected_token_ranks": [1], "position_ids_sha256": "p", "image_span_fingerprint": "s"},
            {"generated_token_ids": [1], "selected_token_log_probabilities": [0.0], "selected_token_ranks": [1], "position_ids_sha256": "p", "image_span_fingerprint": "s"},
        )
        self.helper_calls.extend(["static.generate_complete_row", "dynamic.forward_with_dynamic_arm"])
        # The mask and hook are represented as explicit receipts in the fake
        # seam; real execution uses build_k11_key_removal_mask plus
        # forward_with_dynamic_arm in one model call.
        self.persistent_calls.extend([("Y11", 1), ("Y11", 3)])
        self.helper_calls.append("gradient.run_static_dynamic_gradient_path_audit")
        return {
            "event_id": event.get("gt_owner_id"),
            "status": "valid",
            "parser_called": True,
            "same_forward_mask_and_hook": True,
            "helper_calls": list(self.helper_calls),
            "persistent_calls": list(self.persistent_calls),
        }


def _event_boundary(event: Mapping[str, Any], checkpoint: str) -> int | None:
    pair = event.get("A_B", {}).get(checkpoint, {}) if isinstance(event.get("A_B"), Mapping) else {}
    b = pair.get("B_verified_uncovered") if isinstance(pair, Mapping) else None
    value = b.get("natural_boundary") if isinstance(b, Mapping) else None
    if isinstance(value, Mapping):
        for key in ("index", "row_index", "boundary_index", "token_index"):
            if key in value:
                value = value[key]
                break
    try:
        if isinstance(value, bool):
            return None
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _event_owner(event: Mapping[str, Any], key: str, checkpoint: str) -> str | None:
    value: Any = event.get(key)
    if isinstance(value, Mapping) and checkpoint in value:
        value = value.get(checkpoint)
    if isinstance(value, Mapping):
        for candidate in ("gt_owner_id", "owner_id", "target_owner_id", "id"):
            if value.get(candidate) is not None:
                value = value[candidate]
                break
    if value is None and key in {"target_row_ids", "target_row_owner_id", "gt_owner_id"}:
        pair = event.get("A_B", {}).get(checkpoint, {}) if isinstance(event.get("A_B"), Mapping) else {}
        b = pair.get("B_verified_uncovered", {}) if isinstance(pair, Mapping) else {}
        if isinstance(b, Mapping):
            value = b.get("gt_owner_id", b.get("owner_id", b.get("target_owner_id")))
    if value is None or isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
        return None
    return str(value)


def _event_declared_owner(event: Mapping[str, Any], checkpoint: str) -> str | None:
    """Read the event's declared target without falling back to B evidence."""

    for key in ("target_row_owner_id", "gt_owner_id"):
        value: Any = event.get(key)
        if isinstance(value, Mapping):
            value = value.get(checkpoint)
        if isinstance(value, Mapping):
            for candidate in ("gt_owner_id", "owner_id", "target_owner_id", "id"):
                if value.get(candidate) is not None:
                    value = value[candidate]
                    break
        if value is not None and (
            isinstance(value, (str, bytes)) or not isinstance(value, (Mapping, Sequence))
        ):
            return str(value)
    return None


_BOOLEAN_RECEIPT_FIELDS = frozenset(
    {
        "duplicate",
        "physical_match",
        "same_class",
        "same_parent",
        "source_specific",
        "source_specific_physical_owner_match",
        "strict_complete_row",
        "strict_match",
        "verified",
        "verified_support",
    }
)
_BOOLEAN_RECEIPT_FIELD_PATTERNS = (
    re.compile(r"^(?:is_|[a-z0-9]+_)*duplicate$"),
    re.compile(r"^(?:[a-z0-9]+_)*verified$"),
    re.compile(r"^(?:[a-z0-9]+_)*verified_support$"),
)


def _is_boolean_receipt_field(key: Any) -> bool:
    key_text = str(key).lower()
    return key_text in _BOOLEAN_RECEIPT_FIELDS or any(
        pattern.fullmatch(key_text) is not None
        for pattern in _BOOLEAN_RECEIPT_FIELD_PATTERNS
    )


def _validate_boolean_receipts(value: Any, *, path: str = "event") -> None:
    """Reject stringly-typed owner/duplicate evidence before any actuator call.

    Nullable fields are explicit absence (for example, an inactive checkpoint
    or a pre-support ``B_verified_uncovered`` slot), not a boolean claim.  The
    active pair receipt is validated separately and may not use ``None``.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_boolean_receipt_field(key) and not isinstance(child, (Mapping, list, tuple)):
                if child is not None and not isinstance(child, bool):
                    raise OrchestrationError(f"{path}.{key} must be a boolean receipt")
            _validate_boolean_receipts(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_boolean_receipts(child, path=f"{path}[{index}]")


def _required_nonnegative_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OrchestrationError(f"{path} must be a non-negative integer")
    return int(value)


def _required_sha256(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise OrchestrationError(f"{path} must be a lowercase 64-hex SHA-256 digest")
    return value


def _validate_active_event_receipts(
    event: Mapping[str, Any],
    *,
    checkpoint: str,
) -> tuple[str, bool, str | None, dict[str, Any]]:
    """Validate only the selected checkpoint's A/B eligibility receipt.

    The cohort intentionally carries both checkpoint branches.  A missing or
    indeterminate comparator branch is therefore ignored here, while an active
    ``verified_pair`` must carry the complete paired-owner evidence required by
    the actuators.  Known pre-support statuses are retained as non-scored
    events; unknown statuses remain shard-fatal.
    """

    neutral = {key: value for key, value in event.items() if key not in {"A_B", "checkpoint_status"}}
    _validate_boolean_receipts(neutral)
    status_root = event.get("checkpoint_status")
    status = status_root.get(checkpoint) if isinstance(status_root, Mapping) else None
    if status is not None and not isinstance(status, Mapping):
        raise OrchestrationError(f"event.checkpoint_status.{checkpoint} must be an object")
    if isinstance(status, Mapping):
        _validate_boolean_receipts(status, path=f"event.checkpoint_status.{checkpoint}")

    pairs = event.get("A_B")
    if not isinstance(pairs, Mapping):
        raise OrchestrationError("event.A_B must be an object")
    pair = pairs.get(checkpoint)
    if not isinstance(pair, Mapping):
        raise OrchestrationError(f"event.A_B.{checkpoint} must be an object")
    pair_status = pair.get("pair_status")
    if not isinstance(pair_status, str):
        raise OrchestrationError(f"event.A_B.{checkpoint}.pair_status must be a string")
    target_owner = _event_declared_owner(event, checkpoint)
    if target_owner is None:
        raise OrchestrationError(f"event lacks a target owner for checkpoint {checkpoint}")
    b_receipt = pair.get("B_verified_uncovered")
    if pair_status in KNOWN_INELIGIBLE_PAIR_STATUSES:
        if b_receipt is not None:
            raise OrchestrationError(
                f"event.A_B.{checkpoint}.B_verified_uncovered must be null for {pair_status}"
            )
        return str(target_owner), False, pair_status, dict(pair)
    if pair_status != "verified_pair":
        raise OrchestrationError(f"event.A_B.{checkpoint}.pair_status is unknown: {pair_status!r}")
    if not isinstance(b_receipt, Mapping):
        raise OrchestrationError(f"event.A_B.{checkpoint}.B_verified_uncovered must be a mapping")
    required_b = {"gt_owner_id", "verified_support", "strict_complete_row", "natural_boundary", "exact_prefix_sha256"}
    missing_b = sorted(required_b - set(b_receipt))
    if missing_b:
        raise OrchestrationError(
            f"event.A_B.{checkpoint}.B_verified_uncovered is missing field(s): {','.join(missing_b)}"
        )
    if b_receipt.get("gt_owner_id") != target_owner:
        raise OrchestrationError(
            f"event.A_B.{checkpoint}.B_verified_uncovered.gt_owner_id differs from event target owner"
        )
    if b_receipt.get("verified_support") is not True:
        raise OrchestrationError(f"event.A_B.{checkpoint}.B_verified_uncovered.verified_support must be true")
    if b_receipt.get("strict_complete_row") is not False:
        raise OrchestrationError(f"event.A_B.{checkpoint}.B_verified_uncovered.strict_complete_row must be false")
    b_boundary = _required_nonnegative_int(
        b_receipt.get("natural_boundary"),
        path=f"event.A_B.{checkpoint}.B_verified_uncovered.natural_boundary",
    )
    _required_sha256(
        b_receipt.get("exact_prefix_sha256"),
        path=f"event.A_B.{checkpoint}.B_verified_uncovered.exact_prefix_sha256",
    )

    a_receipt = pair.get("A_latest_covered")
    if not isinstance(a_receipt, Mapping):
        raise OrchestrationError(f"event.A_B.{checkpoint}.A_latest_covered must be a mapping")
    required_a = {"gt_owner_id", "source_panel_object_index", "natural_boundary", "strict_complete_row"}
    missing_a = sorted(required_a - set(a_receipt))
    if missing_a:
        raise OrchestrationError(
            f"event.A_B.{checkpoint}.A_latest_covered is missing field(s): {','.join(missing_a)}"
        )
    a_owner = a_receipt.get("gt_owner_id")
    if not isinstance(a_owner, str) or not a_owner or a_owner == target_owner:
        raise OrchestrationError(f"event.A_B.{checkpoint}.A_latest_covered.gt_owner_id is invalid")
    _required_nonnegative_int(
        a_receipt.get("source_panel_object_index"),
        path=f"event.A_B.{checkpoint}.A_latest_covered.source_panel_object_index",
    )
    a_boundary = _required_nonnegative_int(
        a_receipt.get("natural_boundary"),
        path=f"event.A_B.{checkpoint}.A_latest_covered.natural_boundary",
    )
    if a_receipt.get("strict_complete_row") is not True:
        raise OrchestrationError(f"event.A_B.{checkpoint}.A_latest_covered.strict_complete_row must be true")
    if a_boundary >= b_boundary:
        raise OrchestrationError("active A_latest_covered boundary must be strictly earlier than B boundary")
    return str(target_owner), True, pair_status, {**dict(pair), "_a_owner": a_owner, "_a_boundary": a_boundary, "_b_boundary": b_boundary}


def _row_from_owner(runtime: ImageRuntime, owner_id: str, tokenizer: Any, contract: static.WrapperContract) -> list[int] | None:
    mapping = getattr(runtime, "owner_mapping", {}).get(str(owner_id))
    if not isinstance(mapping, Mapping):
        return None
    derived_index = mapping.get("derived_index")
    if isinstance(derived_index, bool) or not isinstance(derived_index, int):
        return None
    objects = _raw_objects(runtime.raw)
    if derived_index < 0 or derived_index >= len(objects):
        return None
    obj = objects[derived_index]
    description = str(
        _raw_field(
            obj,
            "description",
            _raw_field(obj, "desc", _raw_field(obj, "category_name", _raw_field(obj, "category", ""))),
        )
    )
    desc_ids = tokenizer.encode(description, add_special_tokens=False)
    bins = _raw_coordinate_bins(runtime.raw, obj)
    if bins is None:
        return None
    return [
        contract.object_ref_start_token_id,
        *[int(v) for v in desc_ids],
        contract.object_ref_end_token_id,
        contract.box_start_token_id,
        *[contract.coordinate_token_start_id + int(v) for v in bins],
        contract.box_end_token_id,
        *([contract.commit_token_id] if contract.commit_token_id is not None else []),
    ]


def _geometry_actuator_gate(event: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Require the materializer's verified launch disposition before actuators."""

    if event.get("geometry_launch_eligible") is not True:
        return False, "geometry_launch_eligible is not true"
    if event.get("geometry_status") != "available":
        return False, f"geometry_status is {event.get('geometry_status')!r}, expected 'available'"
    if event.get("geometry_mechanical_disposition") != "eligible_verified_pair_regions":
        return False, "geometry mechanical disposition is not eligible_verified_pair_regions"
    regions = event.get("image_cell_regions")
    if not isinstance(regions, Mapping):
        return False, "verified image_cell_regions are absent"
    for key in ("a_exclusive", "b_exclusive", "background"):
        values = regions.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            return False, f"required geometry region {key} is empty or malformed"
    return True, None


def _strict_materialization_geometry_gate(
    event: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Classify geometry only when the frozen receipt is conclusive.

    The live actuator gate intentionally treats every non-eligible shape as a
    safe negative.  Contract materialization has a narrower job: it may omit
    an event only when the cohort positively says that geometry is either
    eligible or not applicable.  Missing, indeterminate, or novel values are
    source drift and therefore shard-fatal.
    """

    launch_eligible = event.get("geometry_launch_eligible")
    if not isinstance(launch_eligible, bool):
        raise OrchestrationError(
            "materialization geometry_launch_eligible must be a boolean receipt"
        )
    geometry_status = event.get("geometry_status")
    disposition = event.get("geometry_mechanical_disposition")
    eligible, reason = _geometry_actuator_gate(event)
    if launch_eligible:
        if not eligible:
            raise OrchestrationError(
                "materialization geometry receipt drift: " + str(reason)
            )
        return True, None
    if geometry_status != "not_applicable" or disposition != "not_applicable":
        raise OrchestrationError(
            "materialization geometry receipt drift: an ineligible receipt must "
            "use geometry_status='not_applicable' and "
            "geometry_mechanical_disposition='not_applicable'"
        )
    return False, "geometry receipt is conclusively not applicable"


def _partition_ineligible_materialization_shard(
    events: Sequence[Mapping[str, Any]],
    *,
    checkpoint: Literal["S", "A"],
    event_shard: tuple[int, int],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Shard original cohort indices first, then retain only safe negatives."""

    shard, count = event_shard
    if count != 4:
        raise OrchestrationError("ineligible materialization requires modulo-4 event sharding")
    original_shard = [
        (index, event)
        for index, event in enumerate(events)
        if index % count == shard
    ]
    selected: list[Mapping[str, Any]] = []
    selected_indices: list[int] = []
    excluded: list[dict[str, Any]] = []
    for cohort_index, event in original_shard:
        target_owner, pair_eligible, _pair_status, _pair = _validate_active_event_receipts(
            event,
            checkpoint=checkpoint,
        )
        geometry_eligible, _geometry_reason = _strict_materialization_geometry_gate(event)
        receipt = {
            "cohort_index": cohort_index,
            "cohort_ordinal": cohort_index + 1,
            "event_id": target_owner,
            "image_id": str(event.get("image_id")),
        }
        if pair_eligible and geometry_eligible:
            excluded.append(receipt)
            continue
        selected.append(event)
        selected_indices.append(cohort_index)
    return selected, {
        "selection_order": "original_cohort_enumerate_then_modulo4_then_filter",
        "original_cohort_event_count": len(events),
        "event_shard": {"shard": shard, "count": count},
        "original_shard_cohort_indices": [index for index, _event in original_shard],
        "original_shard_cohort_ordinals": [index + 1 for index, _event in original_shard],
        "materialized_cohort_indices": selected_indices,
        "materialized_cohort_ordinals": [index + 1 for index in selected_indices],
        "excluded_actuator_eligible_events": excluded,
        "original_shard_event_count": len(original_shard),
        "materialized_event_count": len(selected),
        "excluded_actuator_eligible_event_count": len(excluded),
    }


def _bind_materialization_event_h0_context(
    adapter: ExperimentRuntimeAdapter,
    event: Mapping[str, Any],
) -> None:
    """Project one exact ledger prefix onto the existing context constructor.

    Cohort ``natural_boundary`` counts decision boundaries, whereas the raw H0
    trace stores every generated row.  The live experiment historically sees
    only the active eligible boundary.  Non-scored enumeration spans all
    decision boundaries, so bind the ledger's exact token prefix explicitly
    after proving it is a literal prefix of the immutable H0 token trace.
    """

    image_id = str(event.get("image_id"))
    target_owner = _event_declared_owner(event, adapter.checkpoint)
    if target_owner is None:
        raise OrchestrationError(
            f"materialization event lacks target owner for checkpoint {adapter.checkpoint}"
        )
    runtime = adapter.panel_rows.get(image_id)
    trace = adapter.h0_rows.get(image_id)
    ledger = adapter.h0_ledger_records.get(image_id, {}).get(str(target_owner))
    if runtime is None or trace is None or ledger is None:
        raise OrchestrationError(
            f"materialization lacks runtime/H0 ledger binding for {image_id}/{target_owner}"
        )
    exact_prefix = tuple(int(value) for value in ledger.get("exact_prefix_token_ids", ()))
    generated = tuple(int(value) for value in trace.get("generated_token_ids", ()))
    if generated[: len(exact_prefix)] != exact_prefix:
        raise OrchestrationError(
            f"materialization ledger prefix is not an exact H0 trace prefix for {image_id}/{target_owner}"
        )
    if sha256_token_ids(exact_prefix) != ledger.get("exact_prefix_sha256"):
        raise OrchestrationError(
            f"materialization ledger prefix hash drift for {image_id}/{target_owner}"
        )
    boundary = ledger.get("natural_boundary")
    if isinstance(boundary, bool) or not isinstance(boundary, int) or boundary < 0:
        raise OrchestrationError(
            f"materialization natural boundary is invalid for {image_id}/{target_owner}"
        )
    contextual_rows: list[list[int]] = []
    if boundary > 0:
        contextual_rows = [*([[]] * (boundary - 1)), list(exact_prefix)]
    runtime.h0 = {
        **dict(trace),
        "rows": contextual_rows,
        "materialization_trace_binding": {
            "source": "checkpoint_native_h0_ledger_exact_prefix",
            "trace_sha256": trace.get("trace_sha256"),
            "generated_token_ids_sha256": trace.get("generated_token_ids_sha256"),
            "exact_prefix_sha256": ledger.get("exact_prefix_sha256"),
            "exact_prefix_token_count": len(exact_prefix),
            "natural_boundary": boundary,
            "literal_trace_prefix": True,
        },
    }


def _history_resolution_value(
    resolution: Any,
    keys: Sequence[str],
    *,
    label: str,
    required: bool = True,
) -> Any:
    """Read one field from a resolver receipt without accepting loose tuples."""

    source: Mapping[str, Any]
    if isinstance(resolution, Mapping):
        source = resolution
    else:
        try:
            source = vars(resolution)
        except TypeError as exc:
            raise OrchestrationError(
                f"history resolver must return a mapping/dataclass receipt, not {type(resolution).__name__}"
            ) from exc
    for key in keys:
        if key in source:
            return source[key]
    if required:
        raise OrchestrationError(
            f"history resolver receipt lacks {label} ({'/'.join(keys)})"
        )
    return None


def _normalize_history_resolution(resolution: Any) -> dict[str, Any]:
    """Normalize the narrow injected history-resolver contract."""

    history_raw = _history_resolution_value(
        resolution,
        (
            "exact_history_token_ids",
            "history_token_ids",
            "b_history_token_ids",
            "b_history",
        ),
        label="exact B history token IDs",
    )
    history_sha = _history_resolution_value(
        resolution,
        ("exact_history_sha256", "history_sha256", "b_history_sha256"),
        label="exact B history SHA-256",
    )
    a_history_raw = _history_resolution_value(
        resolution,
        (
            "exact_A_history_token_ids",
            "exact_a_history_token_ids",
            "a_history_token_ids",
            "a_history",
            "exact_A_history",
            "covered_A_history_token_ids",
        ),
        label="exact A history token IDs",
    )
    a_history_sha = _history_resolution_value(
        resolution,
        (
            "exact_A_history_sha256",
            "exact_a_history_sha256",
            "a_history_sha256",
            "a_history_hash",
            "exact_A_history_hash",
            "covered_A_history_sha256",
        ),
        label="exact A history SHA-256",
    )
    completed_raw = _history_resolution_value(
        resolution,
        ("completed_row_ids", "physical_completed_prefix_rows", "completed_rows"),
        label="physical completed prefix rows",
    )
    latest_raw = _history_resolution_value(
        resolution,
        (
            "latest_row_ids",
            "latest_covered_row_ids",
            "latest_covered_physical_row",
            "latest_row",
            "latest_physical_row",
        ),
        label="latest covered physical row",
    )

    def token_ids(value: Any, *, label: str, allow_empty: bool = True) -> tuple[int, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise OrchestrationError(f"history resolver {label} must be an integer sequence")
        result: list[int] = []
        for token in value:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise OrchestrationError(
                    f"history resolver {label} contains an invalid token ID"
                )
            result.append(int(token))
        if not allow_empty and not result:
            raise OrchestrationError(f"history resolver {label} must not be empty")
        return tuple(result)

    exact_history = token_ids(history_raw, label="exact B history token IDs", allow_empty=False)
    exact_a_history = token_ids(a_history_raw, label="exact A history token IDs")
    latest = token_ids(latest_raw, label="latest covered physical row", allow_empty=False)
    if isinstance(completed_raw, (str, bytes)) or not isinstance(completed_raw, Sequence):
        raise OrchestrationError(
            "history resolver physical completed prefix rows must be a sequence"
        )
    completed: tuple[tuple[int, ...], ...] = tuple(
        token_ids(row, label="physical completed prefix row", allow_empty=False)
        for row in completed_raw
    )
    if not completed:
        raise OrchestrationError("history resolver physical completed prefix rows are empty")
    if not isinstance(history_sha, str) or not history_sha:
        raise OrchestrationError("history resolver exact B history SHA-256 is malformed")
    if not isinstance(a_history_sha, str) or not a_history_sha:
        raise OrchestrationError("history resolver exact A history SHA-256 is malformed")
    receipt = _history_resolution_value(
        resolution,
        ("receipt", "resolver_receipt"),
        label="resolver receipt",
        required=False,
    )
    if receipt is None:
        receipt = {}
    if not isinstance(receipt, Mapping):
        raise OrchestrationError("history resolver receipt must be a mapping")
    return {
        "exact_history_token_ids": exact_history,
        "exact_history_sha256": history_sha,
        "exact_A_history_token_ids": exact_a_history,
        "exact_A_history_sha256": a_history_sha,
        "completed_row_ids": completed,
        "latest_row_ids": latest,
        "receipt": dict(receipt),
    }


def _make_event_context(
    adapter: ExperimentRuntimeAdapter,
    event: Mapping[str, Any],
    *,
    history_resolver: Callable[[ExperimentRuntimeAdapter, Mapping[str, Any]], Any] | None = None,
) -> EventContext:
    target_owner_hint, pair_actuator_eligible, pair_status, active_pair = _validate_active_event_receipts(
        event,
        checkpoint=adapter.checkpoint,
    )
    geometry_eligible, geometry_reason = _geometry_actuator_gate(event)
    actuator_eligible = bool(pair_actuator_eligible and geometry_eligible)
    eligibility_reason = (
        None
        if actuator_eligible
        else (
            f"active pair is not actuator-eligible: {pair_status}"
            if not pair_actuator_eligible
            else f"geometry is not actuator-eligible: {geometry_reason}"
        )
    )
    image_id = str(event.get("image_id"))
    runtime = adapter.panel_rows.get(image_id)
    if runtime is None:
        raise OrchestrationError(f"cohort event image {image_id} is absent from derived panel/H0")
    target_owner = target_owner_hint
    ledger_for_image = adapter.h0_ledger_records.get(image_id, {})
    ledger_record = ledger_for_image.get(target_owner)
    if ledger_record is None:
        raise OrchestrationError(
            f"event target owner {target_owner} has no checkpoint-native H0 boundary record for {image_id}"
        )
    ledger_boundary = ledger_record.get("natural_boundary")
    event_boundary = _event_boundary(event, adapter.checkpoint)
    status = event.get("checkpoint_status", {}).get(adapter.checkpoint, {}) if isinstance(event.get("checkpoint_status"), Mapping) else {}
    status_boundary = status.get("natural_boundary") if isinstance(status, Mapping) else None
    if event_boundary is not None and event_boundary != ledger_boundary:
        raise OrchestrationError(
            f"event boundary {event_boundary} differs from checkpoint-native H0 boundary {ledger_boundary} "
            f"for {image_id}/{target_owner}"
        )
    if isinstance(status_boundary, bool) or (status_boundary is not None and int(status_boundary) != ledger_boundary):
        raise OrchestrationError(
            f"checkpoint status boundary {status_boundary} differs from checkpoint-native H0 boundary {ledger_boundary} "
            f"for {image_id}/{target_owner}"
        )
    if not isinstance(ledger_boundary, int) or isinstance(ledger_boundary, bool) or ledger_boundary < 0:
        raise OrchestrationError(f"checkpoint-native H0 boundary is invalid for {image_id}/{target_owner}")
    boundary = ledger_boundary
    rows = runtime.h0.get("rows", [])
    if boundary > len(rows):
        raise OrchestrationError(f"event boundary {boundary} is outside H0 row count {len(rows)} for {image_id}")
    resolved_history: dict[str, Any] | None = None
    if history_resolver is not None:
        if not callable(history_resolver):
            raise OrchestrationError("history_resolver must be callable or None")
        try:
            resolved_history = _normalize_history_resolution(
                history_resolver(adapter, event)
            )
        except OrchestrationError:
            raise
        except Exception as exc:
            raise OrchestrationError(
                f"history resolver failed for {image_id}/{target_owner}: {exc}"
            ) from exc
    # ``natural_boundary`` is a row boundary; first prove that the exact native
    # generated history agrees with the checkpoint ledger, then append the
    # native row opener for the free row.
    if resolved_history is None:
        history_ids = [int(token) for row in rows[:boundary] for token in row]
    else:
        history_ids = list(resolved_history["exact_history_token_ids"])
    expected_history_ids = ledger_record.get("exact_prefix_token_ids")
    if tuple(history_ids) != tuple(expected_history_ids or ()):
        raise OrchestrationError(
            f"H0 exact generated-history prefix differs from ledger for {image_id}/{target_owner}"
        )
    history_sha256 = sha256_token_ids(history_ids)
    if history_sha256 != ledger_record.get("exact_prefix_sha256"):
        raise OrchestrationError(
            f"H0 exact generated-history prefix hash differs from ledger for {image_id}/{target_owner}"
        )
    if (
        resolved_history is not None
        and resolved_history["exact_history_sha256"] != history_sha256
    ):
        raise OrchestrationError(
            f"history resolver exact B history hash differs from literal history for {image_id}/{target_owner}"
        )
    known_owner_ids = {str(owner.get("owner_id")) for owner in runtime.owners}
    if pair_actuator_eligible:
        b_hash = str(active_pair.get("B_verified_uncovered", {}).get("exact_prefix_sha256"))
        if b_hash != ledger_record.get("exact_prefix_sha256"):
            raise OrchestrationError(
                f"active B exact-prefix hash differs from checkpoint-native H0 ledger for {image_id}/{target_owner}"
            )
        a_owner = active_pair.get("_a_owner")
        a_boundary = active_pair.get("_a_boundary")
        a_record = ledger_for_image.get(str(a_owner))
        if str(a_owner) not in known_owner_ids:
            raise OrchestrationError(f"active A owner {a_owner} is absent from source owner ledger for {image_id}")
        if isinstance(getattr(runtime, "owner_mapping", None), Mapping) and str(a_owner) not in runtime.owner_mapping:
            raise OrchestrationError(f"active A owner {a_owner} is absent from source/derived mapping for {image_id}")
        if a_record is None:
            raise OrchestrationError(f"active A owner {a_owner} has no checkpoint-native H0 ledger record for {image_id}")
        if a_record.get("natural_boundary") != a_boundary:
            raise OrchestrationError(
                f"active A boundary differs from checkpoint-native H0 ledger for {image_id}/{a_owner}"
            )
        if a_record.get("strict_complete_row") is not None and a_record.get("strict_complete_row") is not True:
            raise OrchestrationError(f"active A ledger row is not strict-complete for {image_id}/{a_owner}")
        if resolved_history is None:
            a_history_ids = [int(token) for row in rows[: int(a_boundary)] for token in row]
        else:
            a_history_ids = list(resolved_history["exact_A_history_token_ids"])
        if tuple(a_history_ids) != tuple(a_record.get("exact_prefix_token_ids") or ()):
            raise OrchestrationError(f"active A exact generated-history prefix differs from ledger for {image_id}/{a_owner}")
        if sha256_token_ids(a_history_ids) != a_record.get("exact_prefix_sha256"):
            raise OrchestrationError(f"active A exact generated-history prefix hash differs from ledger for {image_id}/{a_owner}")
        if (
            resolved_history is not None
            and resolved_history["exact_A_history_sha256"]
            != sha256_token_ids(a_history_ids)
        ):
            raise OrchestrationError(
                f"history resolver exact A history hash differs from literal history for {image_id}/{a_owner}"
            )
    prefix = [int(value) for value in runtime.prompt_ids.detach().cpu().reshape(-1).tolist()]
    prefix.extend(history_ids)
    prefix.append(adapter.wrapper_contract.object_ref_start_token_id)
    if resolved_history is None:
        latest = list(rows[boundary - 1]) if boundary > 0 else []
        completed_row_ids = tuple(
            tuple(int(token) for token in row) for row in rows[:boundary]
        )
    else:
        latest = list(resolved_history["latest_row_ids"])
        completed_row_ids = tuple(resolved_history["completed_row_ids"])
        if tuple(token for row in completed_row_ids for token in row) != tuple(history_ids):
            raise OrchestrationError(
                f"history resolver completed physical rows do not flatten to exact B history for {image_id}/{target_owner}"
            )
        if tuple(latest) != tuple(completed_row_ids[-1]):
            raise OrchestrationError(
                f"history resolver latest physical row is not the completed-prefix suffix for {image_id}/{target_owner}"
            )
    if target_owner not in known_owner_ids:
        raise OrchestrationError(f"event target owner {target_owner} is absent from source owner ledger for {image_id}")
    mapping = getattr(runtime, "owner_mapping", {}).get(target_owner)
    if isinstance(getattr(runtime, "owner_mapping", None), Mapping):
        if not isinstance(mapping, Mapping):
            raise OrchestrationError(f"event target owner {target_owner} has no source/derived mapping for {image_id}")
        panel_identity = event.get("panel_identity")
        if not isinstance(panel_identity, Mapping):
            raise OrchestrationError(f"event target owner {target_owner} lacks panel_identity for {image_id}")
        expected_panel_identity = {
            "source_panel_object_index": mapping.get("source_index"),
            "derived_panel_object_index": mapping.get("derived_index"),
            "coco_ann_id": mapping.get("coco_ann_id"),
        }
        for key, expected in expected_panel_identity.items():
            observed = panel_identity.get(key)
            if key == "coco_ann_id":
                if str(observed) != str(expected):
                    raise OrchestrationError(
                        f"event panel identity {key} differs from source/derived mapping for {image_id}/{target_owner}"
                    )
            elif observed != expected:
                raise OrchestrationError(
                    f"event panel identity {key} differs from source/derived mapping for {image_id}/{target_owner}"
                )
        if panel_identity.get("status") not in {None, "matched"}:
            raise OrchestrationError(f"event panel identity is not matched for {image_id}/{target_owner}")
    target_row = event.get("target_row_token_ids")
    if isinstance(target_row, Mapping):
        target_row = target_row.get(adapter.checkpoint)
    mapped_target_ids = None
    if isinstance(mapping, Mapping):
        mapped_target_ids = _row_from_owner(runtime, target_owner, adapter.tokenizer, adapter.wrapper_contract)
        if not mapped_target_ids:
            raise OrchestrationError(f"source/derived mapping cannot generate target row for {image_id}/{target_owner}")
    if isinstance(target_row, Sequence) and not isinstance(target_row, (str, bytes)):
        target_ids = [int(v) for v in target_row]
        if mapped_target_ids is not None and target_ids != mapped_target_ids:
            raise OrchestrationError(f"event target row differs from mapped derived owner row for {image_id}/{target_owner}")
    else:
        target_ids = mapped_target_ids or _row_from_owner(runtime, target_owner, adapter.tokenizer, adapter.wrapper_contract)
    if not target_ids:
        raise OrchestrationError(f"event {event.get('gt_owner_id')} lacks a target row for P4")
    # The cohort's A_latest_covered field is a convenience label, not the
    # complete parent history.  Bind the full ordered owner set from the
    # checkpoint-native H0 ledger at this exact natural boundary so G/K/L and
    # repeat classification cannot silently collapse to the latest owner.
    covered = list(ledger_record.get("covered_owner_ids", ()))
    unknown_covered = [value for value in covered if value not in known_owner_ids]
    if unknown_covered:
        raise OrchestrationError(f"event covered owner IDs are absent from source owner ledger: {unknown_covered}")
    if isinstance(getattr(runtime, "owner_mapping", None), Mapping):
        unknown_mapped = [value for value in covered if value not in runtime.owner_mapping]
        if unknown_mapped:
            raise OrchestrationError(f"event covered owner IDs are absent from source/derived mapping: {unknown_mapped}")
    latest_covered = ledger_record.get("latest_covered_owner_id")
    if covered and latest_covered != covered[-1]:
        raise OrchestrationError(f"checkpoint-native H0 covered-owner order is inconsistent for {image_id}/{target_owner}")
    prefix_receipt = {
        "target_owner_id": target_owner,
        "natural_boundary": boundary,
        "covered_owner_ids": list(covered),
        "event_eligibility": {
            "status": "eligible" if actuator_eligible else "invalid/uninterpretable",
            "pair_status": pair_status,
            "geometry_status": event.get("geometry_status"),
            "geometry_mechanical_disposition": event.get("geometry_mechanical_disposition"),
            "geometry_launch_eligible": event.get("geometry_launch_eligible"),
            "reason": eligibility_reason,
        },
        "h0": {
            "exact_generated_history_prefix_sha256": history_sha256,
            "exact_generated_history_token_count": len(history_ids),
            "ledger_source_path": ledger_record.get("source_path"),
            "ledger_record_index": ledger_record.get("record_index"),
            "exact_prefix_token_ids": list(history_ids),
            "trace_binding": runtime.h0.get("materialization_trace_binding"),
        },
        "model_input": {
            "prefix_sha256": sha256_token_ids(prefix),
            "prefix_token_count": len(prefix),
            "prefix_token_ids": list(prefix),
            "row_opener_token_id": adapter.wrapper_contract.object_ref_start_token_id,
            "wrapper_mode": "commit" if adapter.wrapper_contract.commit_token_id is not None else "closed",
        },
    }
    if resolved_history is not None:
        prefix_receipt["h0"]["history_resolver"] = dict(resolved_history["receipt"])
    support_ids: list[str] = []
    support_measured = False
    for candidate_owner, candidate_record in ledger_for_image.items():
        if not isinstance(candidate_record, Mapping):
            continue
        value = candidate_record.get("verified_support")
        if value is True:
            support_measured = True
            support_ids.append(str(candidate_owner))
        elif value is False:
            support_measured = True
    prefix_receipt["support_contract"] = (
        {
            "status": "measured",
            "source": "checkpoint_native_h0_support_records",
            "owner_ids": sorted(set(support_ids)),
            "binding": "same-checkpoint-same-prefix-support-ledger",
        }
        if support_measured
        else _not_measured("checkpoint-native ledger has no verified support field")
    )
    if isinstance(mapping, Mapping):
        prefix_receipt["panel_identity"] = {
            "source_panel_object_index": mapping.get("source_index"),
            "derived_panel_object_index": mapping.get("derived_index"),
            "coco_ann_id": mapping.get("coco_ann_id"),
            "mapping_method": mapping.get("mapping_method"),
            "mapping_receipt_sha256": getattr(runtime, "mapping_receipt", {}).get("mapping_sha256"),
        }
        mapping_receipt = getattr(runtime, "mapping_receipt", {})
        prefix_receipt["owner_mapping"] = {
            "schema_version": mapping_receipt.get("schema_version"),
            "image_id": mapping_receipt.get("image_id"),
            "mapping_sha256": mapping_receipt.get("mapping_sha256"),
            "source_owner_count": mapping_receipt.get("source_owner_count"),
            "derived_owner_count": mapping_receipt.get("derived_owner_count"),
            "mapping_method_census": mapping_receipt.get("mapping_method_census", {}),
            "source_to_derived": mapping_receipt.get("source_to_derived", []),
        }
    return EventContext(
        event=event,
        runtime=runtime,
        prefix_ids=torch.tensor([prefix], dtype=torch.long, device=adapter.model_device),
        natural_boundary=boundary,
        latest_row_ids=latest,
        covered_owner_ids=tuple(covered),
        target_row_ids=target_ids,
        uncovered_row_ids=target_ids,
        prefix_receipt=prefix_receipt,
        actuator_eligible=actuator_eligible,
        eligibility_reason=eligibility_reason,
        pair_status=pair_status,
        completed_row_ids=completed_row_ids,
        checkpoint=adapter.checkpoint,
    )


def _ineligible_event_result(
    context: EventContext,
    *,
    checkpoint: str,
    runtime_attestation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Emit a complete non-scored event matrix without touching the model."""

    reason = context.eligibility_reason or "active cohort pair is not actuator-eligible"
    def invalid(label: str) -> dict[str, str]:
        return {"status": "invalid/uninterpretable", "reason": f"{label}: {reason}"}
    p1 = {probe: invalid(probe) for probe in P1_ARM_IDS}
    p2 = {arm: invalid(arm) for arm in dynamic.DYNAMIC_ARM_IDS}
    p3 = {cell: invalid(cell) for cell in dynamic.P3_CELL_IDS}
    return {
        "event_id": context.event.get("gt_owner_id"),
        "image_id": context.runtime.image_id,
        "checkpoint": checkpoint,
        "runtime_attestation": None if runtime_attestation is None else dict(runtime_attestation),
        "eligibility": {
            "status": "invalid/uninterpretable",
            "pair_status": context.pair_status,
            "reason": reason,
            "actuators_called": False,
        },
        "prefix": dict(context.prefix_receipt),
        "p1": {"status": "invalid/uninterpretable", "reason": reason, "arms": p1},
        "p2": {"status": "invalid/uninterpretable", "reason": reason, "arms": p2},
        "p3": {"status": "invalid/uninterpretable", "reason": reason, "cells": p3},
        "p4": {"status": "invalid/uninterpretable", "reason": reason},
    }


def _query_physical_gpu_identity(physical_device_id: str) -> dict[str, Any]:
    """Resolve one visible CUDA token to a physical index and UUID."""

    if re.fullmatch(r"[0-9]+", physical_device_id) is None:
        raise OrchestrationError("CUDA_VISIBLE_DEVICES must expose exactly one numeric device token")
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise OrchestrationError("nvidia-smi is unavailable for physical GPU attestation")
    try:
        completed = subprocess.run(
            [
                executable,
                "-i",
                physical_device_id,
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OrchestrationError(f"physical GPU attestation query failed: {exc}") from exc
    if completed.returncode != 0:
        raise OrchestrationError("physical GPU attestation query returned non-zero")
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise OrchestrationError("physical GPU attestation query did not return exactly one device")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 2 or fields[0] != physical_device_id:
        raise OrchestrationError("physical GPU attestation index disagrees with CUDA_VISIBLE_DEVICES")
    uuid = fields[1]
    if re.fullmatch(r"GPU-[0-9A-Fa-f-]+", uuid) is None:
        raise OrchestrationError("physical GPU attestation UUID is missing or malformed")
    return {
        "physical_device_id": physical_device_id,
        "physical_device_index": int(physical_device_id),
        "physical_device_uuid": uuid,
    }


def _attest_live_runtime(adapter: ExperimentRuntimeAdapter) -> dict[str, Any]:
    """Capture support-probe-aligned logical and physical runtime identity."""

    config = adapter.config
    expected_dtype = str(getattr(getattr(config, "model", None), "dtype", "fp32"))
    expected_attention = str(
        getattr(getattr(getattr(config, "backend", None), "hf", None), "attn_implementation", "sdpa")
    )
    try:
        runtime = support.validate_live_runtime_identity(
            adapter.session,
            expected_dtype=expected_dtype,
            expected_attention=expected_attention,
            expected_checkpoint=adapter.checkpoint,
            expected_config_fingerprint=str(adapter.resolved.fingerprint),
        )
        visible = support._cuda_visible_mapping()  # noqa: SLF001 - reuse the support contract exactly
    except Exception as exc:
        raise OrchestrationError(f"live runtime identity validation failed: {exc}") from exc
    tokens = visible.get("tokens") if isinstance(visible, Mapping) else None
    if not isinstance(tokens, list) or len(tokens) != 1 or not isinstance(tokens[0], str):
        raise OrchestrationError("CUDA_VISIBLE_DEVICES must expose exactly one numeric device token")
    physical_device_id = tokens[0].strip()
    if re.fullmatch(r"[0-9]+", physical_device_id) is None:
        raise OrchestrationError("CUDA_VISIBLE_DEVICES must expose exactly one numeric device token")
    visible = {**dict(visible), "tokens": [physical_device_id], "selected_physical_device": physical_device_id}
    process_id = os.getpid()
    physical = _query_physical_gpu_identity(physical_device_id)
    first_parameter = next(adapter.model.parameters(), None)
    if first_parameter is None:
        raise OrchestrationError("live runtime model has no parameters for device attestation")
    first_parameter_device = str(first_parameter.device)
    model_device = str(adapter.model_device)
    normalized = runtime.get("normalized_device")
    if first_parameter_device != model_device or first_parameter_device != normalized:
        raise OrchestrationError("live runtime logical device differs across model/current/first-parameter identity")
    try:
        properties = torch.cuda.get_device_properties(int(normalized.rsplit(":", 1)[-1]))
        properties_uuid = getattr(properties, "uuid", None)
    except Exception as exc:  # pragma: no cover - live CUDA boundary
        raise OrchestrationError(f"live runtime physical GPU properties are unavailable: {exc}") from exc
    if properties_uuid is None:
        raise OrchestrationError("live runtime physical GPU UUID is absent from torch properties")
    physical_uuid_raw = str(physical["physical_device_uuid"])
    torch_uuid_raw = str(properties_uuid)
    physical_uuid_normalized = physical_uuid_raw.lower().removeprefix("gpu-")
    torch_uuid_normalized = torch_uuid_raw.lower().removeprefix("gpu-")
    if re.fullmatch(r"[0-9a-f-]+", torch_uuid_normalized) is None or physical_uuid_normalized != torch_uuid_normalized:
        raise OrchestrationError("live runtime physical GPU UUID differs between torch and nvidia-smi")
    physical["physical_device_uuid_raw"] = physical_uuid_raw
    physical["physical_device_uuid_normalized"] = physical_uuid_normalized
    physical["torch_device_uuid_raw"] = torch_uuid_raw
    return {
        "schema_version": RUNTIME_ATTESTATION_SCHEMA_VERSION,
        "status": "validated",
        "passed": True,
        "checkpoint": adapter.checkpoint,
        "config_fingerprint": str(adapter.resolved.fingerprint),
        "device": runtime["device"],
        "effective_device": runtime["effective_device"],
        "normalized_device": runtime["normalized_device"],
        "logical_selected_device": runtime["normalized_device"],
        "model_device": model_device,
        "torch_current_device": runtime["torch_current_device"],
        "first_parameter_device": first_parameter_device,
        "cuda_visible_devices": visible,
        **physical,
        "pid": process_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "support_runtime_identity": runtime,
    }


def _not_applicable_runtime_attestation(*, reason: str) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_ATTESTATION_SCHEMA_VERSION,
        "status": "not_applicable",
        "passed": False,
        "reason": reason,
    }


def _native_row_entry_margin(
    adapter: ExperimentRuntimeAdapter,
    context: EventContext,
) -> dict[str, Any]:
    """Score opener versus native STOP at the exact pre-opener boundary."""

    contract = adapter.wrapper_contract
    if contract.eos_token_id is None:
        return _not_measured("native tokenizer has no <|im_end|>/EOS token")
    prefix = context.prefix_ids
    if not isinstance(prefix, torch.Tensor) or prefix.ndim != 2 or int(prefix.shape[1]) < 2:
        return _not_measured("exact natural prefix has no pre-opener boundary")
    boundary_ids = prefix[:, :-1]
    payload, _position_ids, _mrope = adapter.exact_model_inputs(context.runtime, boundary_ids)
    with torch.inference_mode():
        output = adapter.model(**payload)
    raw_logits = output.logits if hasattr(output, "logits") else output.get("logits") if isinstance(output, Mapping) else None
    if not isinstance(raw_logits, torch.Tensor) or raw_logits.ndim != 3:
        return _not_measured("native row-entry scalar forward did not expose [batch,sequence,vocab] logits")
    logits = raw_logits[0, -1].detach().float()
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all().item()):
        return _not_measured("native row-entry scalar forward returned non-finite logits")
    values = torch.log_softmax(logits, dim=-1)
    row_entry = values[contract.object_ref_start_token_id]
    terminal = values[contract.eos_token_id]
    if not bool(torch.isfinite(torch.stack((row_entry, terminal))).all().item()):
        return _not_measured("native row-entry scalar log-probabilities are non-finite")
    return {
        "status": "measured",
        "source": "exact_native_pre_opener_scalar_forward",
        "teacher_forced": False,
        "row_entry_token_id": int(contract.object_ref_start_token_id),
        "terminal_token_id": int(contract.eos_token_id),
        "row_entry_log_probability": float(row_entry.item()),
        "terminal_log_probability": float(terminal.item()),
        "row_entry_minus_terminal": float((row_entry - terminal).item()),
        "prefix_without_opener_sha256": sha256_token_ids(boundary_ids),
    }


def _support_owner_contract(context: EventContext) -> dict[str, Any]:
    """Resolve independently verified same-image support without guessing."""

    event = getattr(context, "event", {})
    if not isinstance(event, Mapping):
        event = {}
    for key in (
        "verified_support_owner_ids",
        "independently_verified_support_owner_ids",
        "support_owner_ids",
    ):
        value = event.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            owners = tuple(sorted({str(item) for item in value if isinstance(item, (str, int))}))
            if owners:
                return {
                    "status": "measured",
                    "source": f"event.{key}",
                    "owner_ids": list(owners),
                    "binding": "event-owned-independent-support",
                }
            return _not_measured(f"event.{key} is empty; no independently verified support claim")
    context_prefix_receipt = getattr(context, "prefix_receipt", {})
    bound = context_prefix_receipt.get("support_contract") if isinstance(context_prefix_receipt, Mapping) else None
    if isinstance(bound, Mapping):
        return dict(bound)
    records = getattr(context.runtime, "h0_ledger_records", None)
    # Runtime rows normally carry the image-local ledger through the adapter;
    # injected tests may instead put it directly on the image runtime.
    if not isinstance(records, Mapping):
        records = getattr(context.runtime, "support_records", None)
    if isinstance(records, Mapping):
        support_ids: list[str] = []
        measured = False
        for owner_id, record in records.items():
            if not isinstance(record, Mapping):
                continue
            support = record.get("verified_support")
            if support is True:
                measured = True
                support_ids.append(str(owner_id))
            elif support is False:
                measured = True
        if measured:
            return {
                "status": "measured",
                "source": "checkpoint_native_h0_support_records",
                "owner_ids": sorted(set(support_ids)),
                "binding": "same-checkpoint-same-prefix-support-ledger",
            }
    # The adapter owns the authoritative image->owner ledger.  Keep this
    # fallback explicit because a missing support field is not a scientific
    # empty support set.
    adapter_records = getattr(context, "h0_ledger_records", None)
    if isinstance(adapter_records, Mapping):
        return _not_measured("support records were not attached to the event context")
    return _not_measured("no independently verified same-image support record is available")


def _owner_match_status(row_result: Mapping[str, Any]) -> tuple[str, str | None, Mapping[str, Any]]:
    parsed = row_result.get("native_parse")
    match = row_result.get("owner_match")
    if not isinstance(match, Mapping):
        match = parsed.get("owner_match", {}) if isinstance(parsed, Mapping) else {}
    owner_id = match.get("owner_id", match.get("matched_owner_id")) if isinstance(match, Mapping) else None
    status = str(match.get("status", "unmatched")) if isinstance(match, Mapping) else "unmatched"
    return status, None if owner_id is None else str(owner_id), match


def _endpoint_row_evidence(
    context: EventContext,
    row_result: Mapping[str, Any],
    *,
    active_intervention: str,
    contract: static.WrapperContract | None = None,
    row_index: int = 0,
    seen_owner_ids: Sequence[str] = (),
    stop_observed: bool | None = None,
) -> dict[str, Any]:
    """Seal one natural P1/P2/P3 row with endpoint and identity evidence."""

    generated = row_result.get("generated_token_ids")
    generated_tokens = (
        [int(value) for value in generated]
        if isinstance(generated, Sequence) and not isinstance(generated, (str, bytes))
        else []
    )
    native_parse = row_result.get("native_parse")
    parsed = row_result.get("parsed")
    parse_status = (
        str(native_parse.get("parse_status"))
        if isinstance(native_parse, Mapping) and native_parse.get("parse_status") is not None
        else str(parsed.get("parse_status")) if isinstance(parsed, Mapping) and parsed.get("parse_status") is not None else "not_measured"
    )
    match_status, owner_id, match = _owner_match_status(row_result)
    strict_accepted = bool(
        isinstance(native_parse, Mapping)
        and native_parse.get("valid") is True
        and parse_status == "accepted"
        and match_status in {"unique", "matched"}
        and match.get("source_specific") is not False
        and match.get("physical_match") is not False
        and owner_id is not None
    )
    prior = {str(value) for value in seen_owner_ids}
    if not prior:
        prior.update(str(value) for value in getattr(context, "covered_owner_ids", ()))
    duplicate = owner_id is not None and owner_id in prior
    newly_covered = [owner_id] if strict_accepted and owner_id is not None and owner_id not in prior else []

    stop_reason = row_result.get("stop_reason")
    if stop_reason is None and isinstance(row_result.get("evidence_row"), Mapping):
        stop_reason = row_result["evidence_row"].get("stop_reason")
    if stop_reason is not None:
        stop_reason = str(stop_reason)
    closure_ids = set()
    if contract is not None:
        closure_ids.add(contract.closure_token_id)
        if contract.eos_token_id is not None:
            closure_ids.add(contract.eos_token_id)
    # The live wrapper contract is included by the caller after this helper;
    # parse-level detection remains conservative when no stop receipt exists.
    closure_positions = [index for index, token in enumerate(generated_tokens) if token in closure_ids]
    over_continuation = bool(closure_positions and closure_positions[0] != len(generated_tokens) - 1)
    premature_stop = stop_reason in {"eos", "im_end", "terminal"} and not strict_accepted
    maxed = stop_reason in {"max_new_tokens", "length", "horizon_exhausted"}
    if over_continuation:
        premature_stop = False

    support = _support_owner_contract(context)
    prefix_receipt = getattr(context, "prefix_receipt", {})
    if not isinstance(prefix_receipt, Mapping):
        prefix_receipt = {}
    remaining: dict[str, Any]
    if support.get("status") != "measured":
        remaining = _not_measured("remaining independently verified support cannot be bound at this natural row")
    else:
        covered_now = prior | set(newly_covered)
        remaining_ids = sorted(set(support.get("owner_ids", ())) - covered_now)
        if stop_observed is True:
            remaining = {
                "status": "measured",
                "owner_ids": remaining_ids,
                "count": len(remaining_ids),
                "support_source": support.get("source"),
                "at_stop": True,
            }
        else:
            remaining = {
                "status": "measured",
                "owner_ids": remaining_ids,
                "count": len(remaining_ids),
                "support_source": support.get("source"),
                "at_stop": False,
            }

    selected_logs = row_result.get("selected_token_log_probabilities")
    selected_ranks = row_result.get("selected_token_ranks")
    # Segment arithmetic is completed once the live caller supplies the native
    # wrapper contract.  Keep a provenance slot here even when the candidate
    # score is unavailable for this arm.
    segment_evidence: dict[str, Any] = _not_measured(
        "native wrapper contract was not supplied to endpoint evidence helper"
    )
    if isinstance(row_result.get("endpoint_segments"), Mapping):
        segment_evidence = dict(row_result["endpoint_segments"])
    elif contract is not None:
        row_tokens = row_result.get("native_parse", {}).get("row_token_ids") if isinstance(row_result.get("native_parse"), Mapping) else None
        if isinstance(row_tokens, Sequence) and not isinstance(row_tokens, (str, bytes)):
            segment_evidence = _natural_segment_scores(
                row_tokens,
                selected_logs,
                selected_ranks,
                contract=contract,
            )
    row_entry = row_result.get("row_entry_vs_terminal")
    if not isinstance(row_entry, Mapping):
        row_entry = _not_measured(
            "row-entry versus native <|im_end|> requires an exact boundary scalar forward for this arm"
        )
    active_scores = row_result.get("active_endpoint_scores")
    if isinstance(active_scores, Mapping):
        active_row_entry = active_scores.get("row_entry_vs_native_im_end")
        if isinstance(active_row_entry, Mapping):
            row_entry = dict(active_row_entry)
    active_target_deltas = (
        active_scores.get("comparisons", {}).get("target_deltas", {})
        if isinstance(active_scores, Mapping)
        and isinstance(active_scores.get("comparisons"), Mapping)
        and isinstance(active_scores.get("comparisons", {}).get("target_deltas"), Mapping)
        else {}
    )
    uncovered_deltas = {
        str(label): value
        for label, value in active_target_deltas.items()
        if str(label).startswith("verified-uncovered:")
        and isinstance(value, Mapping)
        and value.get("status") == "measured"
        and isinstance(value.get("candidate_minus_target_mean_log_probability"), Mapping)
        and isinstance(
            value.get("candidate_minus_target_mean_log_probability", {}).get("full_row"),
            (int, float),
        )
    }
    if uncovered_deltas:
        best_uncovered_label = max(
            uncovered_deltas,
            key=lambda label: float(
                uncovered_deltas[label]["candidate_minus_target_mean_log_probability"]["full_row"]
            ),
        )
        best_uncovered_delta: Mapping[str, Any] = {
            **dict(uncovered_deltas[best_uncovered_label]),
            "best_verified_uncovered_owner": best_uncovered_label.split(":", 1)[1],
        }
    else:
        best_uncovered_delta = _not_measured(
            "no independently verified-uncovered candidate row is available"
        )

    non_target = row_result.get("non_target_damage")
    if not isinstance(non_target, Mapping):
        operator = row_result.get("operator_receipt")
        if isinstance(operator, Mapping) and operator.get("non_target_max_abs_delta") is not None:
            try:
                non_target = {
                    "status": "measured",
                    "kind": "mechanical_non_target_state_drift",
                    "max_abs_delta": _finite_number(
                        operator.get("non_target_max_abs_delta"), label="non-target drift"
                    ),
                }
            except OrchestrationError:
                non_target = _not_measured("non-target drift receipt is non-finite")
        else:
            dynamic_receipts = row_result.get("dynamic_receipts")
            if isinstance(dynamic_receipts, Sequence):
                values = [item.get("non_target_max_abs_delta") for item in dynamic_receipts if isinstance(item, Mapping)]
                if values:
                    try:
                        non_target = {
                            "status": "measured",
                            "kind": "mechanical_non_target_state_drift",
                            "max_abs_delta": max(_finite_number(value, label="non-target drift") for value in values),
                        }
                    except OrchestrationError:
                        non_target = _not_measured("dynamic non-target drift receipt is non-finite")
    if not isinstance(non_target, Mapping):
        non_target = _not_measured("active arm did not expose a non-target drift receipt")

    return {
        "schema_version": ENDPOINT_SCHEMA_VERSION,
        "status": "measured",
        "natural": True,
        "teacher_forced": False,
        "active_intervention": str(active_intervention),
        "row_index": int(row_index),
        "strict_native_endpoint": {
            "status": "accepted" if strict_accepted else "not_accepted",
            "native_parse_status": parse_status,
            "source_specific_physical_owner_match": bool(strict_accepted),
            "owner_match_status": match_status,
            "owner_id": owner_id,
            "target_owner_id": str(
                getattr(context, "event", {}).get("gt_owner_id", "")
                if isinstance(getattr(context, "event", {}), Mapping)
                else ""
            ),
        },
        "scores": {
            "natural_segments": segment_evidence,
            "active_candidate_scores": active_scores
            if isinstance(active_scores, Mapping)
            else _not_measured(
                "no exact active-intervention candidate scorer was attached to this natural release"
            ),
            "target_B_vs_covered_A": (
                active_scores.get("comparisons", {}).get("target_deltas", {}).get("covered-A")
                if isinstance(active_scores, Mapping)
                and isinstance(active_scores.get("comparisons"), Mapping)
                and isinstance(active_scores.get("comparisons", {}).get("target_deltas"), Mapping)
                else _not_measured(
                    "no exact active-intervention target-B/covered-A comparison was attached"
                )
            ),
            "target_B_vs_best_verified_uncovered": (
                best_uncovered_delta
            ),
            "row_entry_vs_native_im_end": row_entry,
            "score_role": "diagnostic_only; never natural behavior",
        },
        "outcome": {
            "valid_row": bool(strict_accepted),
            "duplicate": bool(duplicate),
            "unmatched": bool(match_status == "unmatched"),
            "ambiguous": bool(match_status == "ambiguous"),
            "malformed": bool(parse_status in {"malformed", "invalid"}),
            "generated_token_count": len(generated_tokens),
            "stop_reason": stop_reason if stop_reason is not None else _not_measured("natural stop reason missing"),
            "premature_stop": premature_stop if stop_reason is not None else _not_measured("natural stop reason missing"),
            "over_continuation": over_continuation if stop_reason is not None else _not_measured("natural stop reason missing"),
            "max_new_tokens_reached": maxed if stop_reason is not None else _not_measured("natural stop reason missing"),
        },
        "covered_A_repeat": bool(owner_id is not None and owner_id in set(getattr(context, "covered_owner_ids", ()))),
        "non_target_damage": non_target,
        "newly_covered_unique_owner_ids": newly_covered,
        "remaining_independently_verified_support_at_stop": remaining,
        "identity_binding": {
            "event_id": getattr(context, "event", {}).get("gt_owner_id") if isinstance(getattr(context, "event", {}), Mapping) else None,
            "image_id": getattr(context.runtime, "image_id", None),
            "checkpoint": getattr(context, "checkpoint", None),
            "natural_prefix_sha256": prefix_receipt.get("model_input", {}).get("prefix_sha256") if isinstance(prefix_receipt.get("model_input"), Mapping) else None,
            "h0_exact_prefix_sha256": prefix_receipt.get("h0", {}).get("exact_generated_history_prefix_sha256") if isinstance(prefix_receipt.get("h0"), Mapping) else None,
            "covered_owner_ids": list(getattr(context, "covered_owner_ids", ())),
            "support": support,
            "active_arm_binding": str(active_intervention),
        },
    }


def _release_row_with_static(
    adapter: ExperimentRuntimeAdapter,
    context: EventContext,
    *,
    arm: str,
    regions: Mapping[str, Sequence[int]] | None = None,
    residual_layer: int | None = None,
    residual_arm: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_RELEASE_TOKENS,
) -> dict[str, Any]:
    runtime = context.runtime
    calls: list[dict[str, Any]] = []
    residual_factory: Callable[..., Any] | None = None
    if residual_layer is not None and residual_arm is not None:
        layer, _ = static.resolve_decoder_layer(adapter.model, residual_layer)
        if regions is None:
            raise OrchestrationError(f"{residual_arm} requires declared event image-cell regions")
        a_indices = list(regions.get("a_exclusive", regions.get("covered_a_exclusive", ())))
        target_indices = list(regions.get("b_exclusive", ()))
        background_indices = list(regions.get("background", ()))
        shared_indices = list(regions.get("shared_core", regions.get("shared", ())))
        if residual_arm == "R00":
            target_indices = list(range(runtime.image_span.token_count))
        if residual_arm == "R10" and (not target_indices or not background_indices):
            raise OrchestrationError("R10 lacks equal-count target/background image cells")
        if residual_arm == "R10":
            # Equal-count selection is deterministic and part of the receipt;
            # a differing region census never changes the destination arity.
            count = min(len(target_indices), len(background_indices))
            target_indices = sorted(target_indices)[:count]
            background_indices = sorted(background_indices)[:count]
        if residual_arm == "R11":
            if not a_indices or not target_indices:
                raise OrchestrationError("R11 lacks non-empty A/B exclusive image cells")
            count = min(len(a_indices), len(target_indices))
            a_indices = sorted(a_indices)[:count]
            b_indices = sorted(target_indices)[:count]
            target_indices = [*a_indices, *b_indices]

        def make_context(*, step: int, input_ids: torch.Tensor, position_ids: torch.Tensor, span: static.ImageSpan) -> Any:
            captured = static.capture_post_block_image_field(
                adapter.model,
                layer_idx=residual_layer,
                span=span,
                input_ids=input_ids,
                model_inputs=runtime.native_inputs,
                position_ids=position_ids,
            )
            state = captured["state"]
            if residual_arm == "R12":
                selected = shared_indices
                if not selected:
                    raise OrchestrationError("R12 requires shared-core image cells")
                source = state[selected]
                replacement, receipt = static.build_residual_replacement(
                    arm="R12", target_state=source, shared_state=source
                )
            else:
                source = state[target_indices]
                background = state[background_indices]
                donor_state = (
                    torch.cat((state[b_indices], state[a_indices]), dim=0)
                    if residual_arm == "R11"
                    else None
                )
                replacement, receipt = static.build_residual_replacement(
                    arm=residual_arm, target_state=source,
                    background_state=background if residual_arm == "R10" else None,
                    donor_state=donor_state,
                )
            calls.append({
                "step": int(step),
                "capture": captured["receipt"],
                "replacement": receipt,
                "selected_target_indices": list(target_indices),
                "selected_background_indices": list(background_indices),
                "selected_a_indices": list(a_indices),
                "selected_b_indices": list(b_indices) if residual_arm == "R11" else list(regions.get("b_exclusive", ())),
            })
            return static.MultiPositionResidualReplacement(
                layer,
                absolute_positions=static.resolve_image_positions(span, target_indices if residual_arm != "R12" else shared_indices),
                replacement=replacement,
                remove_after_first=True,
                operator_arm=residual_arm,
            )

        residual_factory = make_context

    result = static.generate_complete_row(
        adapter.model,
        prefix_ids=context.prefix_ids,
        image_token_id=runtime.image_token_id,
        image_grid_thw=runtime.image_grid_thw,
        merge_size=runtime.merge_size,
        runtime_contract=adapter.wrapper_contract,
        arm=arm, regions=regions,
        model_inputs=runtime.native_inputs,
        max_new_tokens=max_new_tokens,
        tokenizer=adapter.tokenizer,
        residual_factory=residual_factory,
        residual_arm=residual_arm,
    )
    actuator_status = result.get("status")
    if actuator_status == "not_applicable":
        return result
    if actuator_status not in {None, "complete", "incomplete"}:
        raise OrchestrationError(f"{arm} actuator returned unsupported status {actuator_status!r}")
    parsed = adapter.parse_row(result.get("parsed", {}).get("row_token_ids", [adapter.wrapper_contract.object_ref_start_token_id, *result.get("generated_token_ids", [])]), runtime, row_index=0)
    # A completed native actuator call is a valid observation even when the
    # emitted row is malformed or has no unique physical owner match.  Those
    # are scientific negative/ambiguous outcomes, not mechanical failures.
    if actuator_status is not None:
        result["generation_status"] = actuator_status
    result["status"] = "valid"
    result["native_parse"] = parsed
    result["owner_match"] = parsed.get("owner_match")
    result["residual_factory_calls"] = calls
    if arm == "K00" and residual_arm is None:
        result["row_entry_vs_terminal"] = _native_row_entry_margin(adapter, context)
    else:
        result["row_entry_vs_terminal"] = _not_measured(
            "exact pre-opener scalar forward was not run for active intervention arm"
        )
    effective_arm = str(arm)
    if residual_arm is not None:
        effective_arm = f"{residual_arm}_block{int(residual_layer)}"
    def active_forward(ids: torch.Tensor) -> tuple[Any, Mapping[str, Any]]:
        runtime_local = context.runtime
        custom_mask: torch.Tensor | None = None
        if arm != "K00":
            span = static.derive_image_span(
                ids,
                image_token_id=runtime_local.image_token_id,
                image_grid_thw=runtime_local.image_grid_thw,
                merge_size=runtime_local.merge_size,
            )
            relative = static._arm_relative_indices(arm, span=span, regions=regions)
            if relative is None:
                raise OrchestrationError(f"{arm} has no declared active image-cell region")
            custom_mask = static.build_row_query_image_key_mask(
                sequence_length=int(ids.shape[1]),
                image_key_positions=span.absolute_positions,
                eligible_image_positions=static.resolve_image_positions(span, relative),
                query_position=int(ids.shape[1] - 1),
                device=ids.device,
            )
        payload, position_ids, _mrope = adapter.exact_model_inputs(
            runtime_local, ids, attention_mask=custom_mask
        )
        auxiliary_forward_count = 0
        residual_context: Any = None
        if residual_arm is not None:
            if residual_layer is None or regions is None:
                raise OrchestrationError(f"{residual_arm} scorer lacks residual layer/regions")
            span = static.derive_image_span(
                ids,
                image_token_id=runtime_local.image_token_id,
                image_grid_thw=runtime_local.image_grid_thw,
                merge_size=runtime_local.merge_size,
            )
            layer, _ = static.resolve_decoder_layer(adapter.model, residual_layer)
            captured = static.capture_post_block_image_field(
                adapter.model,
                layer_idx=residual_layer,
                span=span,
                input_ids=ids,
                model_inputs=runtime_local.native_inputs,
                position_ids=position_ids,
                custom_mask=custom_mask,
            )
            state = captured["state"]
            a_indices = list(regions.get("a_exclusive", regions.get("covered_a_exclusive", ())))
            target_indices = list(regions.get("b_exclusive", ()))
            background_indices = list(regions.get("background", ()))
            shared_indices = list(regions.get("shared_core", regions.get("shared", ())))
            if residual_arm == "R00":
                target_indices = list(range(runtime_local.image_span.token_count))
            if residual_arm == "R10":
                count = min(len(target_indices), len(background_indices))
                if count <= 0:
                    raise OrchestrationError("R10 scorer lacks equal-count target/background cells")
                target_indices, background_indices = sorted(target_indices)[:count], sorted(background_indices)[:count]
            if residual_arm == "R11":
                count = min(len(a_indices), len(target_indices))
                if count <= 0:
                    raise OrchestrationError("R11 scorer lacks non-empty A/B exclusive cells")
                a_indices = sorted(a_indices)[:count]
                b_indices = sorted(target_indices)[:count]
                target_indices = [*a_indices, *b_indices]
            if residual_arm == "R12":
                if not shared_indices:
                    raise OrchestrationError("R12 scorer lacks shared-core cells")
                target_indices = shared_indices
            target_state = state[target_indices]
            replacement, _replacement_receipt = static.build_residual_replacement(
                arm=residual_arm,
                target_state=target_state,
                background_state=state[background_indices] if residual_arm == "R10" else None,
                donor_state=torch.cat((state[b_indices], state[a_indices]), dim=0) if residual_arm == "R11" else None,
                shared_state=state[shared_indices] if residual_arm == "R12" else None,
            )
            residual_context = static.MultiPositionResidualReplacement(
                layer,
                absolute_positions=static.resolve_image_positions(
                    span, target_indices if residual_arm != "R12" else shared_indices
                ),
                replacement=replacement,
                remove_after_first=True,
                operator_arm=residual_arm,
            )
            auxiliary_forward_count = 1
        with nullcontext() if residual_context is None else residual_context:
            with torch.inference_mode():
                output = adapter.model(**payload)
        return output, {
            "active_intervention": effective_arm,
            "intervention_applied": bool(arm != "K00" or residual_arm is not None),
            "source": "active_intervention_scalar_forward",
            "auxiliary_forward_count": auxiliary_forward_count,
        }

    result["active_endpoint_scores"] = _score_active_endpoint_candidates(
        adapter,
        context,
        active_intervention=effective_arm,
        forward_fn=active_forward,
        contract=adapter.wrapper_contract,
    )
    result["endpoint_evidence"] = _endpoint_row_evidence(
        context,
        result,
        active_intervention=effective_arm,
        contract=adapter.wrapper_contract,
    )
    return result


def _capture_dynamic_replacement(
    adapter: ExperimentRuntimeAdapter,
    runtime: ImageRuntime,
    ids: torch.Tensor,
    positions: Sequence[int],
    *,
    background_positions: Sequence[int] = (),
    arm_id: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Capture one dynamic destination and build its declared replacement.

    D01 is an exact numeric self replacement.  D10--D12 mute one carrier by
    replacing it with a norm-matched mean of non-target states from the same
    native row.  D20 applies the corresponding whole-row mean; D21 is
    intentionally not synthesized here because its donor must come from an
    independently proven same-parent row.
    """

    layer, _ = dynamic.resolve_block23(adapter.model)
    target_positions = tuple(int(value) for value in positions)
    background_positions = tuple(int(value) for value in background_positions)
    if not target_positions or len(set(target_positions)) != len(target_positions):
        raise OrchestrationError(f"{arm_id} requires unique dynamic destination positions")
    if set(target_positions) & set(background_positions):
        raise OrchestrationError(f"{arm_id} dynamic target/background positions overlap")
    capture_positions = (*target_positions, *background_positions)
    capture = dynamic.ResidualSpanCapture(layer, capture_positions)
    payload, position_ids, _mrope = adapter.exact_model_inputs(runtime, ids)
    with capture:
        with torch.inference_mode():
            adapter.model(**payload)
    if capture.state is None:
        raise OrchestrationError("block23 dynamic capture returned no state")
    target_state = capture.state[: len(target_positions)]
    background_state = capture.state[len(target_positions) :]
    if arm_id == "D01":
        if not torch.equal(target_state, target_state.detach()):
            raise OrchestrationError("D01 self replacement failed numeric identity")
        replacement = target_state.detach().clone()
        operation = "exact_self"
    elif arm_id in {"D10", "D11", "D12"}:
        if not background_positions:
            raise OrchestrationError(f"{arm_id} requires non-target same-row background states")
        replacement = dynamic.norm_matched_span(target_state, background_state)
        operation = "norm_matched_same_row_background"
    elif arm_id == "D20":
        # Whole-row mute uses one norm-matched row-mean vector at every
        # destination position.  This is distinct from the single-carrier
        # controls and preserves the attested row span/arity.
        replacement = dynamic.norm_matched_span(target_state, target_state)
        operation = "norm_matched_whole_row_mean"
    elif arm_id == "D21":
        raise OrchestrationError("D21 requires an independently proven same-parent donor row")
    else:
        raise OrchestrationError(f"unsupported dynamic replacement arm {arm_id}")
    if not bool(torch.isfinite(replacement).all().item()):
        raise OrchestrationError(f"{arm_id} replacement is non-finite")
    numeric_self = bool(torch.allclose(replacement, target_state, atol=NOOP_TOLERANCE, rtol=0.0))
    if arm_id != "D01" and numeric_self:
        raise OrchestrationError(f"{arm_id} replacement degenerates to the target self state")
    receipt = {
        "arm": arm_id,
        "operation": operation,
        "target_positions": list(target_positions),
        "background_positions": list(background_positions),
        "target_state_sha256": sha256_tensor(target_state),
        "background_state_sha256": sha256_tensor(background_state) if background_positions else None,
        "replacement_state_sha256": sha256_tensor(replacement),
        "target_shape": list(target_state.shape),
        "background_shape": list(background_state.shape),
        "numeric_self": numeric_self,
        "capture_call_count": int(capture.call_count),
        "hook_clean": capture.handle is None,
    }
    return replacement, receipt


def _release_dynamic_row(
    adapter: ExperimentRuntimeAdapter,
    context: EventContext,
    *,
    arm_id: str,
    latest_row_ids: Sequence[int],
    static_regions: Mapping[str, Sequence[int]] | None = None,
    p3_cell_id: str | None = None,
    persistent: bool = False,
    max_new_tokens: int = DEFAULT_MAX_RELEASE_TOKENS,
) -> dict[str, Any]:
    runtime = context.runtime
    current = context.prefix_ids.detach().clone()
    generated: list[int] = []
    selected_log_probs: list[float] = []
    selected_ranks: list[int] = []
    top_prediction_token_ids: list[int] = []
    receipts: list[dict[str, Any]] = []
    stop_reason = "max_new_tokens"
    carrier = dynamic.build_carrier_contract(
        "commit" if adapter.wrapper_contract.commit_token_id is not None else "closed",
        carrier_token_id=adapter.wrapper_contract.closure_token_id,
        closure_token_id=adapter.wrapper_contract.box_end_token_id,
        wrapper=adapter.wrapper_contract.assistant_format,
    )
    # Freeze dynamic destinations at the exact natural boundary.  As greedy
    # tokens are appended, only the query/mask shape grows; the carrier/row
    # positions must remain attached to the completed natural history row.
    frozen_prefix_width = int(current.shape[1] - 1 - len(latest_row_ids))
    if frozen_prefix_width < 0:
        raise OrchestrationError(f"{arm_id} latest row is not contained in the current prefix")
    earlier_row_ids: Sequence[int] | None = None
    frozen_earlier_prefix_width = frozen_prefix_width
    frozen_positions: tuple[int, ...] = ()
    frozen_background_positions: tuple[int, ...] = ()
    if arm_id == "D12":
        completed_rows = getattr(context, "completed_row_ids", ())
        if not completed_rows:
            completed_rows = tuple(
                tuple(int(token) for token in row)
                for row in runtime.h0.get("rows", [])[: context.natural_boundary]
            )
        if len(completed_rows) < 2:
            raise OrchestrationError("D12 requires an earlier equal-length completed row")
        if tuple(int(token) for token in latest_row_ids) != tuple(completed_rows[-1]):
            raise OrchestrationError("D12 latest row differs from the current accepted horizon history")
        earlier_row_ids = completed_rows[-2]
        frozen_earlier_prefix_width = frozen_prefix_width - len(earlier_row_ids)
        if frozen_earlier_prefix_width < 0:
            raise OrchestrationError("D12 earlier row is not contained in the exact prefix")
    if arm_id != "D00":
        if not latest_row_ids:
            raise OrchestrationError(f"{arm_id} requires a prior complete native row")
        row_for_position = (
            earlier_row_ids if arm_id == "D12" and earlier_row_ids is not None else latest_row_ids
        )
        row_prefix_width = (
            frozen_earlier_prefix_width if arm_id == "D12" else frozen_prefix_width
        )
        if arm_id in {"D01", "D10", "D12"}:
            positions = dynamic.row_carrier_position(
                row_for_position,
                prefix_width=row_prefix_width,
                carrier=carrier,
            )
        elif arm_id == "D11":
            positions = dynamic.last_coordinate_position(
                latest_row_ids, prefix_width=frozen_prefix_width
            )
        else:
            positions = dynamic.row_span_positions(
                latest_row_ids, prefix_width=frozen_prefix_width
            )
        frozen_positions = (
            tuple(int(value) for value in positions)
            if isinstance(positions, (tuple, list))
            else (int(positions),)
        )
        frozen_arm = dynamic.build_dynamic_arm(
            arm_id,
            prefix_width=row_prefix_width,
            latest_row_token_ids=latest_row_ids,
            earlier_row_token_ids=earlier_row_ids,
            carrier=carrier,
            replacement_persistence="persistent" if persistent else "one_shot",
        )
        if frozen_arm.status != "ready":
            raise OrchestrationError(f"{arm_id} is not applicable: {frozen_arm.reason}")
        row_for_background = (
            earlier_row_ids if arm_id == "D12" and earlier_row_ids is not None else latest_row_ids
        )
        background_prefix_width = (
            frozen_earlier_prefix_width if arm_id == "D12" else frozen_prefix_width
        )
        background_span = dynamic.row_span_positions(
            row_for_background, prefix_width=background_prefix_width
        )
        if arm_id in {"D01", "D10", "D11", "D12"}:
            frozen_background_positions = tuple(
                int(position)
                for position in background_span
                if position not in frozen_positions
            )
    else:
        frozen_arm = dynamic.build_dynamic_arm(
            "D00",
            prefix_width=frozen_prefix_width,
            replacement_persistence="persistent" if persistent else "one_shot",
        )
    for step in range(max_new_tokens):
        payload, position_ids, _position_hash = adapter.exact_model_inputs(runtime, current)
        native_mrope_grid = payload.get("image_grid_thw", runtime.image_grid_thw)
        native_rope_deltas = payload.get("rope_deltas")
        mrope_hash = dynamic.compute_mrope_hash(
            position_ids,
            image_grid_thw=native_mrope_grid,
            rope_deltas=native_rope_deltas,
        )
        # The key mask is rebuilt at every greedy boundary.  Reusing a mask
        # from the initial opener would have the wrong [S,S] shape after the
        # first emitted token and would silently stop applying K11.
        k11_mask: torch.Tensor | None = None
        if static_regions is not None:
            a_positions = static_regions.get("a_exclusive", static_regions.get("covered_a_exclusive", ()))
            if not a_positions:
                raise OrchestrationError("K11 requires non-empty covered-A-exclusive image cells")
            k11_mask = dynamic.build_k11_key_removal_mask(
                sequence_length=int(current.shape[1]),
                image_key_positions=runtime.image_span.absolute_positions,
                a_exclusive_positions=static.resolve_image_positions(runtime.image_span, a_positions),
                query_positions=[int(current.shape[1] - 1)],
                device=current.device,
            )
        if arm_id != "D00":
            replacement, replacement_receipt = _capture_dynamic_replacement(
                adapter,
                runtime,
                current,
                frozen_positions,
                background_positions=frozen_background_positions,
                arm_id=arm_id,
            )
            contract = dynamic.ExactPrefixContract(
                prefix_token_ids=tuple(int(v) for v in current[0].detach().cpu().tolist()),
                position_ids=position_ids,
                mrope_hash=mrope_hash,
                wrapper=adapter.wrapper_contract.assistant_format,
                image_grid_thw=native_mrope_grid,
                rope_deltas=native_rope_deltas,
            )
            model_inputs = dict(payload)
            model_inputs["mrope_hash"] = mrope_hash
            model_inputs["position_ids"] = position_ids
            model_inputs["input_ids"] = current
            if p3_cell_id is not None:
                static_arm = dynamic.build_static_arm(
                    "K11" if static_regions is not None else "K00",
                    sequence_length=int(current.shape[1]) if static_regions is not None else None,
                    image_key_positions=runtime.image_span.absolute_positions if static_regions is not None else (),
                    a_exclusive_positions=static.resolve_image_positions(runtime.image_span, static_regions.get("a_exclusive", static_regions.get("covered_a_exclusive", ()))) if static_regions is not None else (),
                    query_positions=(int(current.shape[1] - 1),) if static_regions is not None else (),
                    device=current.device,
                )
                cell = dynamic.P3Cell(
                    p3_cell_id,
                    static_arm,
                    frozen_arm,
                    "persistent" if persistent else "one_shot",
                )
                output, receipt = dynamic.forward_with_p3_cell(
                    adapter.model,
                    model_inputs,
                    prefix_contract=contract,
                    cell=cell,
                    replacement=replacement,
                    persistent=persistent,
                )
            else:
                output, receipt = dynamic.forward_with_dynamic_arm(
                    adapter.model,
                    model_inputs,
                    prefix_contract=contract,
                    arm=frozen_arm,
                    replacement=replacement,
                    persistent=persistent,
                    attention_mask=k11_mask,
                )
            receipt["step"] = int(step)
            receipt["replacement_contract"] = replacement_receipt
            receipts.append(receipt)
        else:
            # D00 still goes through the helper so a paired K11 mask is
            # applied in the same native forward, with an explicit no-hook
            # receipt rather than a second unmasked model call.
            contract = dynamic.ExactPrefixContract(
                prefix_token_ids=tuple(int(v) for v in current[0].detach().cpu().tolist()),
                position_ids=position_ids,
                mrope_hash=mrope_hash,
                wrapper=adapter.wrapper_contract.assistant_format,
                image_grid_thw=native_mrope_grid,
                rope_deltas=native_rope_deltas,
            )
            model_inputs = dict(payload)
            model_inputs["mrope_hash"] = mrope_hash
            if p3_cell_id is not None:
                static_arm = dynamic.build_static_arm(
                    "K11" if static_regions is not None else "K00",
                    sequence_length=int(current.shape[1]) if static_regions is not None else None,
                    image_key_positions=runtime.image_span.absolute_positions if static_regions is not None else (),
                    a_exclusive_positions=static.resolve_image_positions(runtime.image_span, static_regions.get("a_exclusive", static_regions.get("covered_a_exclusive", ()))) if static_regions is not None else (),
                    query_positions=(int(current.shape[1] - 1),) if static_regions is not None else (),
                    device=current.device,
                )
                output, receipt = dynamic.forward_with_p3_cell(
                    adapter.model,
                    model_inputs,
                    prefix_contract=contract,
                    cell=dynamic.P3Cell(
                        p3_cell_id,
                        static_arm,
                        frozen_arm,
                        "persistent" if persistent else "one_shot",
                    ),
                    persistent=persistent,
                )
            else:
                output, receipt = dynamic.forward_with_dynamic_arm(
                    adapter.model,
                    model_inputs,
                    prefix_contract=contract,
                    arm=frozen_arm,
                    persistent=persistent,
                    attention_mask=k11_mask,
                )
            receipt["step"] = int(step)
            receipts.append(receipt)
        logits = output.logits[0, -1].detach().float()
        log_probs = torch.log_softmax(logits, dim=-1)
        token = int(torch.argmax(logits).item())
        selected_log_probs.append(_finite_number(log_probs[token].item(), label="dynamic selected token log-probability"))
        selected_ranks.append(int(1 + (logits > logits[token]).sum().item()))
        top_prediction_token_ids.append(token)
        generated.append(token)
        current = torch.cat((current, torch.tensor([[token]], dtype=torch.long, device=current.device)), dim=1)
        if token == adapter.wrapper_contract.closure_token_id:
            stop_reason = "commit" if adapter.wrapper_contract.commit_token_id is not None else "box_end"
            break
        if adapter.wrapper_contract.eos_token_id is not None and token == adapter.wrapper_contract.eos_token_id:
            stop_reason = "eos"
            break
    parsed = adapter.wrapper_contract.parse_generated_suffix(generated, tokenizer=adapter.tokenizer)
    row_token_ids = [int(value) for value in parsed.get("row_token_ids", [adapter.wrapper_contract.object_ref_start_token_id, *generated])]
    native = adapter.parse_row(row_token_ids, runtime, row_index=0)
    if receipts and any(item.get("hook_clean") is not True for item in receipts):
        raise OrchestrationError(f"{arm_id} left a dynamic hook installed")
    if arm_id == "D01":
        no_op_drift = max((float(item.get("no_op_max_abs_delta", 0.0)) for item in receipts), default=0.0)
        if no_op_drift > NOOP_TOLERANCE:
            raise OrchestrationError(f"D01 self/no-op replacement drift {no_op_drift} exceeds tolerance")
    else:
        no_op_drift = None
    parser = dynamic.build_native_row_parser(
        "commit" if adapter.wrapper_contract.commit_token_id is not None else "closed",
        object_ref_start_token_id=adapter.wrapper_contract.object_ref_start_token_id,
        object_ref_end_token_id=adapter.wrapper_contract.object_ref_end_token_id,
        box_start_token_id=adapter.wrapper_contract.box_start_token_id,
        box_end_token_id=adapter.wrapper_contract.box_end_token_id,
        coordinate_token_min=adapter.wrapper_contract.coordinate_token_start_id,
        coordinate_token_max=adapter.wrapper_contract.coordinate_token_start_id + adapter.wrapper_contract.coordinate_bin_count - 1,
        coordinate_count=adapter.wrapper_contract.coordinate_count,
        commit_token_id=adapter.wrapper_contract.commit_token_id if adapter.wrapper_contract.commit_token_id is not None else adapter.wrapper_contract.box_end_token_id,
        im_end_token_id=adapter.wrapper_contract.eos_token_id if adapter.wrapper_contract.eos_token_id is not None else dynamic.IM_END_TOKEN_ID,
    )

    def owner_matcher(tokens: tuple[int, ...], _parser_receipt: Mapping[str, Any]) -> Mapping[str, Any]:
        matched = adapter.parse_row(tokens, runtime, row_index=0).get("owner_match", {})
        status = matched.get("status") if isinstance(matched, Mapping) else "unmatched"
        normalized = {"unique": "matched", "unmatched": "unmatched", "ambiguous": "ambiguous"}.get(str(status), "unmatched")
        return {
            "matcher_id": "production.native_physical_owner_match.v1",
            "status": normalized,
            "matched_owner_id": matched.get("owner_id") if normalized == "matched" else None,
            "source_specific": True,
            "physical_match": normalized == "matched",
            "unmatched": normalized == "unmatched",
            "ambiguous": normalized == "ambiguous",
        }

    try:
        evidence_row = dynamic._coerce_row(
            row_token_ids,
            parser=parser,
            owner_matcher=owner_matcher,
            expected_prefix_token_ids=tuple(int(value) for value in context.prefix_ids[0].detach().cpu().tolist()),
            allow_test_adapter=False,
            forward_receipt=receipts[0] if receipts else None,
        )
    except dynamic.TechnicalInvalid as exc:
        raise OrchestrationError(f"{arm_id} row evidence contract failed: {exc}") from exc
    result = {
        "arm": arm_id,
        "generated_token_ids": generated,
        "generated_token_ids_sha256": sha256_token_ids(generated),
        "selected_token_log_probabilities": selected_log_probs,
        "selected_token_ranks": selected_ranks,
        "top_prediction_token_ids": top_prediction_token_ids,
        "stop_reason": stop_reason,
        "row_entry_vs_terminal": (
            _native_row_entry_margin(adapter, context)
            if arm_id == "D00" and static_regions is None and p3_cell_id in {None, "Y00"}
            else _not_measured("exact pre-opener scalar forward was not run for active intervention arm")
        ),
        "closed_at_wrapper": bool(parsed.get("valid")) if isinstance(parsed, Mapping) else False,
        "complete_row": bool(parsed.get("valid")) if isinstance(parsed, Mapping) else False,
        "parsed": parsed,
        "native_parse": native,
        "owner_match": native.get("owner_match"),
        "evidence_row": evidence_row.as_dict(duplicate=False),
        "_evidence_row_obj": evidence_row,
        "dynamic_receipts": receipts,
        "persistent": bool(persistent),
        "same_forward_mask_and_hook": bool(static_regions is not None and arm_id != "D00"),
        "static_mask_applied": bool(static_regions is not None),
        "p3_cell_id": p3_cell_id,
        "no_op_max_abs_delta": no_op_drift,
        "natural": True,
    }
    active_label = str(p3_cell_id) if p3_cell_id is not None else str(arm_id)
    # Candidate scoring shares the same frozen dynamic destination as the
    # natural release.  The scalar input grows, but these absolute positions
    # remain anchored to the completed natural history row.
    scorer_prefix_width = int(context.prefix_ids.shape[1] - 1 - len(latest_row_ids))
    if scorer_prefix_width < 0:
        raise OrchestrationError("active candidate latest row is not contained in the exact prefix")
    scorer_earlier_row_ids: Sequence[int] | None = None
    scorer_earlier_prefix_width = scorer_prefix_width
    scorer_positions: tuple[int, ...] = ()
    scorer_background_positions: tuple[int, ...] = ()
    scorer_arm: Any
    scorer_carrier = dynamic.build_carrier_contract(
        "commit" if adapter.wrapper_contract.commit_token_id is not None else "closed",
        carrier_token_id=adapter.wrapper_contract.closure_token_id,
        closure_token_id=adapter.wrapper_contract.box_end_token_id,
        wrapper=adapter.wrapper_contract.assistant_format,
    )
    if arm_id == "D12":
        completed_rows = getattr(context, "completed_row_ids", ())
        if not completed_rows:
            completed_rows = tuple(
                tuple(int(token) for token in row)
                for row in context.runtime.h0.get("rows", [])[: context.natural_boundary]
            )
        if len(completed_rows) < 2:
            raise OrchestrationError("D12 active candidate scorer lacks an earlier completed row")
        if tuple(int(token) for token in latest_row_ids) != tuple(completed_rows[-1]):
            raise OrchestrationError("D12 active scorer latest row differs from current accepted history")
        scorer_earlier_row_ids = completed_rows[-2]
        scorer_earlier_prefix_width = scorer_prefix_width - len(scorer_earlier_row_ids)
        if scorer_earlier_prefix_width < 0:
            raise OrchestrationError("D12 earlier row is not contained in the exact prefix")
    if arm_id != "D00":
        if not latest_row_ids:
            raise OrchestrationError(f"{arm_id} active candidate scorer requires a prior complete row")
        scorer_row_for_position = (
            scorer_earlier_row_ids
            if arm_id == "D12" and scorer_earlier_row_ids is not None
            else latest_row_ids
        )
        scorer_row_prefix_width = (
            scorer_earlier_prefix_width if arm_id == "D12" else scorer_prefix_width
        )
        if arm_id in {"D01", "D10", "D12"}:
            scorer_raw_positions = dynamic.row_carrier_position(
                scorer_row_for_position,
                prefix_width=scorer_row_prefix_width,
                carrier=scorer_carrier,
            )
        elif arm_id == "D11":
            scorer_raw_positions = dynamic.last_coordinate_position(
                latest_row_ids, prefix_width=scorer_prefix_width
            )
        else:
            scorer_raw_positions = dynamic.row_span_positions(
                latest_row_ids, prefix_width=scorer_prefix_width
            )
        scorer_positions = (
            tuple(int(value) for value in scorer_raw_positions)
            if isinstance(scorer_raw_positions, (tuple, list))
            else (int(scorer_raw_positions),)
        )
        scorer_arm = dynamic.build_dynamic_arm(
            arm_id,
            prefix_width=scorer_row_prefix_width,
            latest_row_token_ids=latest_row_ids,
            earlier_row_token_ids=scorer_earlier_row_ids,
            carrier=scorer_carrier,
            replacement_persistence="persistent" if persistent else "one_shot",
        )
        if scorer_arm.status != "ready":
            raise OrchestrationError(f"{arm_id} active candidate arm is not ready: {scorer_arm.reason}")
        scorer_background_row = (
            scorer_earlier_row_ids
            if arm_id == "D12" and scorer_earlier_row_ids is not None
            else latest_row_ids
        )
        scorer_background_prefix_width = (
            scorer_earlier_prefix_width if arm_id == "D12" else scorer_prefix_width
        )
        scorer_background_span = dynamic.row_span_positions(
            scorer_background_row, prefix_width=scorer_background_prefix_width
        )
        if arm_id in {"D01", "D10", "D11", "D12"}:
            scorer_background_positions = tuple(
                int(position)
                for position in scorer_background_span
                if position not in scorer_positions
            )
    else:
        scorer_arm = dynamic.build_dynamic_arm(
            "D00",
            prefix_width=scorer_prefix_width,
            replacement_persistence="persistent" if persistent else "one_shot",
        )

    def active_forward(ids: torch.Tensor) -> tuple[Any, Mapping[str, Any]]:
        runtime_local = context.runtime
        payload, position_ids, _position_hash = adapter.exact_model_inputs(runtime_local, ids)
        native_mrope_grid = payload.get("image_grid_thw", runtime_local.image_grid_thw)
        native_rope_deltas = payload.get("rope_deltas")
        mrope_hash = dynamic.compute_mrope_hash(
            position_ids,
            image_grid_thw=native_mrope_grid,
            rope_deltas=native_rope_deltas,
        )
        k11_mask: torch.Tensor | None = None
        if static_regions is not None:
            a_positions = static_regions.get("a_exclusive", static_regions.get("covered_a_exclusive", ()))
            if not a_positions:
                raise OrchestrationError("active candidate K11 scorer lacks covered-A-exclusive cells")
            k11_mask = dynamic.build_k11_key_removal_mask(
                sequence_length=int(ids.shape[1]),
                image_key_positions=runtime_local.image_span.absolute_positions,
                a_exclusive_positions=static.resolve_image_positions(runtime_local.image_span, a_positions),
                query_positions=[int(ids.shape[1] - 1)],
                device=ids.device,
            )
        auxiliary_forward_count = 0
        if arm_id != "D00":
            replacement, _replacement_receipt = _capture_dynamic_replacement(
                adapter,
                runtime_local,
                ids,
                scorer_positions,
                background_positions=scorer_background_positions,
                arm_id=arm_id,
            )
            auxiliary_forward_count = 1
            contract = dynamic.ExactPrefixContract(
                prefix_token_ids=tuple(int(value) for value in ids[0].detach().cpu().tolist()),
                position_ids=position_ids,
                mrope_hash=mrope_hash,
                wrapper=adapter.wrapper_contract.assistant_format,
                image_grid_thw=native_mrope_grid,
                rope_deltas=native_rope_deltas,
            )
            model_inputs = dict(payload)
            model_inputs.update({"mrope_hash": mrope_hash, "position_ids": position_ids, "input_ids": ids})
            if p3_cell_id is not None:
                static_arm = dynamic.build_static_arm(
                    "K11" if static_regions is not None else "K00",
                    sequence_length=int(ids.shape[1]) if static_regions is not None else None,
                    image_key_positions=runtime_local.image_span.absolute_positions if static_regions is not None else (),
                    a_exclusive_positions=static.resolve_image_positions(
                        runtime_local.image_span,
                        static_regions.get("a_exclusive", static_regions.get("covered_a_exclusive", ())),
                    ) if static_regions is not None else (),
                    query_positions=(int(ids.shape[1] - 1),) if static_regions is not None else (),
                    device=ids.device,
                )
                output, receipt = dynamic.forward_with_p3_cell(
                    adapter.model,
                    model_inputs,
                    prefix_contract=contract,
                    cell=dynamic.P3Cell(
                        p3_cell_id,
                        static_arm,
                        scorer_arm,
                        "persistent" if persistent else "one_shot",
                    ),
                    replacement=replacement,
                    persistent=persistent,
                )
            else:
                output, receipt = dynamic.forward_with_dynamic_arm(
                    adapter.model,
                    model_inputs,
                    prefix_contract=contract,
                    arm=scorer_arm,
                    replacement=replacement,
                    persistent=persistent,
                    attention_mask=k11_mask,
                )
        else:
            contract = dynamic.ExactPrefixContract(
                prefix_token_ids=tuple(int(value) for value in ids[0].detach().cpu().tolist()),
                position_ids=position_ids,
                mrope_hash=mrope_hash,
                wrapper=adapter.wrapper_contract.assistant_format,
                image_grid_thw=native_mrope_grid,
                rope_deltas=native_rope_deltas,
            )
            model_inputs = dict(payload)
            model_inputs.update({"mrope_hash": mrope_hash, "position_ids": position_ids, "input_ids": ids})
            if p3_cell_id is not None:
                static_arm = dynamic.build_static_arm(
                    "K11" if static_regions is not None else "K00",
                    sequence_length=int(ids.shape[1]) if static_regions is not None else None,
                    image_key_positions=runtime_local.image_span.absolute_positions if static_regions is not None else (),
                    a_exclusive_positions=static.resolve_image_positions(
                        runtime_local.image_span,
                        static_regions.get("a_exclusive", static_regions.get("covered_a_exclusive", ())),
                    ) if static_regions is not None else (),
                    query_positions=(int(ids.shape[1] - 1),) if static_regions is not None else (),
                    device=ids.device,
                )
                output, receipt = dynamic.forward_with_p3_cell(
                    adapter.model,
                    model_inputs,
                    prefix_contract=contract,
                    cell=dynamic.P3Cell(
                        p3_cell_id,
                        static_arm,
                        scorer_arm,
                        "persistent" if persistent else "one_shot",
                    ),
                    persistent=persistent,
                )
            else:
                output, receipt = dynamic.forward_with_dynamic_arm(
                    adapter.model,
                    model_inputs,
                    prefix_contract=contract,
                    arm=scorer_arm,
                    persistent=persistent,
                    attention_mask=k11_mask,
                )
        receipt = dict(receipt)
        receipt.update(
            {
                "active_intervention": active_label,
                "intervention_applied": bool(active_label not in {"D00", "Y00"} or static_regions is not None),
                "source": "active_intervention_scalar_forward",
                "auxiliary_forward_count": auxiliary_forward_count,
            }
        )
        return output, receipt

    result["active_endpoint_scores"] = _score_active_endpoint_candidates(
        adapter,
        context,
        active_intervention=active_label,
        forward_fn=active_forward,
        contract=adapter.wrapper_contract,
    )
    result["endpoint_evidence"] = _endpoint_row_evidence(
        context,
        result,
        active_intervention=(str(p3_cell_id) if p3_cell_id is not None else str(arm_id)),
        contract=adapter.wrapper_contract,
    )
    return result


def _release_dynamic_horizon(
    adapter: ExperimentRuntimeAdapter,
    context: EventContext,
    *,
    arm_id: str,
    static_regions: Mapping[str, Sequence[int]] | None = None,
    p3_cell_id: str | None = None,
    persistent: bool = False,
    max_new_tokens: int = DEFAULT_MAX_RELEASE_TOKENS,
    horizons: Sequence[int] = (1, 3),
) -> dict[str, Any]:
    """Release independent one-row and three-row natural horizons.

    Each horizon starts from the same frozen boundary.  Within a horizon the
    emitted complete row is appended to the prefix before the next row, so
    every dynamic intervention is recomputed at the current absolute
    positions.  P3 marks its hook as persistent for the whole horizon and
    records the per-row receipts; no forced row is ever inserted.
    """

    if arm_id == "D21":
        return {
            "status": "not_applicable",
            "arm": arm_id,
            "reason": "no independently proven same-parent donor row",
            "required_donor_contract": "same-parent complete native row with independent owner identity and matched geometry",
            "persistent": bool(persistent),
            "natural": False,
        }

    result: dict[str, Any] = {}
    for horizon in horizons:
        horizon_value = int(horizon)
        if horizon_value <= 0:
            raise OrchestrationError("dynamic horizon must be positive")
        current_prefix = context.prefix_ids
        latest = list(context.latest_row_ids)
        completed_rows = [
            tuple(int(token) for token in row)
            for row in (
                getattr(context, "completed_row_ids", ())
                if getattr(context, "completed_row_ids", ())
                else context.runtime.h0.get("rows", [])[: context.natural_boundary]
            )
        ]
        rows: list[dict[str, Any]] = []
        evidence_rows: list[Any] = []
        seen_horizon_owner_ids: set[str] = set(context.covered_owner_ids)
        for row_index in range(horizon_value):
            local_context = EventContext(
                event=context.event,
                runtime=context.runtime,
                prefix_ids=current_prefix,
                natural_boundary=len(completed_rows),
                latest_row_ids=latest,
                covered_owner_ids=context.covered_owner_ids,
                target_row_ids=context.target_row_ids,
                uncovered_row_ids=context.uncovered_row_ids,
                prefix_receipt=dict(context.prefix_receipt),
                completed_row_ids=tuple(completed_rows),
                checkpoint=getattr(context, "checkpoint", None),
            )
            row_result = _release_dynamic_row(
                adapter,
                local_context,
                arm_id=arm_id,
                latest_row_ids=latest,
                static_regions=static_regions,
                p3_cell_id=p3_cell_id,
                persistent=persistent,
                max_new_tokens=max_new_tokens,
            )
            row_result["horizon_row_index"] = row_index
            stop_value = str(row_result.get("stop_reason", ""))
            row_result["endpoint_evidence"] = _endpoint_row_evidence(
                local_context,
                row_result,
                active_intervention=(str(p3_cell_id) if p3_cell_id is not None else str(arm_id)),
                contract=adapter.wrapper_contract,
                row_index=row_index,
                seen_owner_ids=tuple(sorted(seen_horizon_owner_ids)),
                stop_observed=stop_value in {"eos", "im_end", "terminal"},
            )
            rows.append(row_result)
            endpoint = row_result.get("endpoint_evidence")
            if isinstance(endpoint, Mapping):
                owner = endpoint.get("strict_native_endpoint", {}).get("owner_id") if isinstance(endpoint.get("strict_native_endpoint"), Mapping) else None
                if isinstance(owner, str) and endpoint.get("outcome", {}).get("valid_row") is True:
                    seen_horizon_owner_ids.add(owner)
            parsed = row_result.get("parsed")
            evidence_object = row_result.pop("_evidence_row_obj", None)
            if isinstance(evidence_object, dynamic.GeneratedRow):
                evidence_rows.append(evidence_object)
            if not isinstance(parsed, Mapping) or not parsed.get("valid"):
                break
            generated = row_result.get("generated_token_ids")
            if not isinstance(generated, Sequence) or isinstance(generated, (str, bytes)):
                break
            # ``generated`` excludes the row opener because the opener is
            # already the final token of the exact prefix.  Append the native
            # closure, then open the next row for a multi-row horizon.
            current_prefix = torch.cat(
                (
                    current_prefix,
                    torch.tensor(
                        [[*map(int, generated), adapter.wrapper_contract.object_ref_start_token_id]],
                        dtype=torch.long,
                        device=current_prefix.device,
                    ),
                ),
                dim=1,
            )
            latest = [adapter.wrapper_contract.object_ref_start_token_id, *map(int, generated)]
            completed_rows.append(tuple(latest))
        try:
            bookkeeping = dynamic.bookkeep_horizon(
                evidence_rows,
                covered_owner_ids=context.covered_owner_ids,
                horizon=horizon_value,
                stop_reason="horizon_exhausted" if len(rows) == horizon_value else "stopped_early",
            )
        except dynamic.TechnicalInvalid as exc:
            raise OrchestrationError(f"{arm_id} horizon owner bookkeeping failed: {exc}") from exc
        result[f"horizon_{horizon_value}"] = {
            "status": "completed" if len(rows) == horizon_value else "stopped_early",
            "rows": rows,
            "row_count": len(rows),
            "persistent": bool(persistent),
            "natural": True,
            "owner_bookkeeping": bookkeeping,
        }
    return result


def _event_regions(event: Mapping[str, Any]) -> Mapping[str, Sequence[int]] | None:
    for key in ("image_cell_regions", "regions", "overlap_regions", "target_regions"):
        value = event.get(key)
        if isinstance(value, Mapping):
            normalized = dict(value)
            if "covered_a_exclusive" not in normalized and "a_exclusive" in normalized:
                normalized["covered_a_exclusive"] = normalized["a_exclusive"]
            if "a_exclusive" not in normalized and "covered_a_exclusive" in normalized:
                normalized["a_exclusive"] = normalized["covered_a_exclusive"]
            if "b_exclusive" not in normalized and "target_b_exclusive" in normalized:
                normalized["b_exclusive"] = normalized["target_b_exclusive"]
            return normalized
    return None


def _verified_owner_regions(
    context: EventContext,
    regions: Mapping[str, Sequence[int]],
) -> tuple[dict[str, tuple[int, ...]], dict[str, dict[str, Any]], str | None]:
    """Bind same-image owner regions and reject shared/overlap leakage."""

    event = context.event
    source = None
    for key in ("owner_regions", "owner_region_cells", "fractional_owner_regions", "regions_by_owner"):
        value = event.get(key)
        if isinstance(value, Mapping):
            source = f"event.{key}"
            owner_values = value
            break
    if source is None:
        return {}, {}, "event lacks independently declared owner-region cells"
    support = _support_owner_contract(context)
    if support.get("status") != "measured":
        return {}, {}, str(support.get("reason", "independent support is not measured"))
    support_ids = {str(owner_id) for owner_id in support.get("owner_ids", ())}
    target_owner = str(context.event.get("gt_owner_id"))
    covered_owner = None
    pairs = context.event.get("A_B")
    if isinstance(pairs, Mapping):
        for pair in pairs.values():
            if not isinstance(pair, Mapping):
                continue
            candidate = pair.get("A_latest_covered")
            if isinstance(candidate, Mapping) and candidate.get("gt_owner_id") is not None:
                covered_owner = str(candidate["gt_owner_id"])
                break
    covered_owner = None if covered_owner is None else str(covered_owner)
    shared_values = regions.get("shared_core", regions.get("shared", ()))
    try:
        shared = {int(value) for value in shared_values}
    except (TypeError, ValueError):
        raise OrchestrationError("P4 shared-core owner region is malformed") from None
    all_cells = set(range(context.runtime.image_span.token_count))
    parsed: dict[str, tuple[int, ...]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    seen: set[int] = set()
    for owner_id in sorted(support_ids):
        raw = owner_values.get(owner_id)
        if raw is None:
            return {}, {}, f"owner-region support is missing for {owner_id}"
        if isinstance(raw, Mapping):
            raw = raw.get("cells", raw.get("indices", raw.get("fractional_cells")))
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise OrchestrationError(f"P4 owner-region cells for {owner_id} are not an integer sequence")
        try:
            cells = tuple(sorted({int(value) for value in raw}))
        except (TypeError, ValueError):
            raise OrchestrationError(f"P4 owner-region cells for {owner_id} are malformed") from None
        if not cells or not set(cells).issubset(all_cells):
            raise OrchestrationError(f"P4 owner-region cells for {owner_id} fall outside the image span")
        if set(cells) & shared:
            raise OrchestrationError(f"P4 owner-region cells for {owner_id} overlap shared-core cells")
        if seen & set(cells):
            raise OrchestrationError(f"P4 owner-region cells overlap another owner ({owner_id})")
        seen.update(cells)
        absolute = tuple(static.resolve_image_positions(context.runtime.image_span, cells))
        parsed[owner_id] = absolute
        receipts[owner_id] = {
            "status": "measured",
            "source": source,
            "fractional_cell_indices": list(cells),
            "absolute_positions": list(absolute),
            "positions_sha256": gradient.sha256_int_sequence(absolute),
            "shared_core_excluded": True,
            "role": "target-B" if owner_id == target_owner else "covered-A" if owner_id == covered_owner else "non-target-owner",
        }
    if target_owner not in parsed:
        return {}, {}, "target-B owner region is absent from the verified support region map"
    if covered_owner is not None and covered_owner not in parsed:
        return {}, {}, "covered-A owner region is absent from the verified support region map"
    # The target state retained by the P4 objective must be exactly the
    # declared B-exclusive region; silently substituting another owner's cells
    # would turn a target gradient into an attribution error.
    declared_target = regions.get("b_exclusive", regions.get("target_b_exclusive", ()))
    try:
        declared_target_positions = tuple(static.resolve_image_positions(context.runtime.image_span, declared_target))
    except (TypeError, ValueError) as exc:
        raise OrchestrationError(f"P4 target-B owner region cannot be resolved: {exc}") from exc
    if tuple(parsed[target_owner]) != tuple(declared_target_positions):
        raise OrchestrationError("P4 target-B owner region differs from declared b_exclusive cells")
    return parsed, receipts, None


def _aligned_row_segment_scores(
    logits: torch.Tensor,
    row_token_ids: Sequence[int],
    *,
    contract: static.WrapperContract,
) -> dict[str, Any]:
    """Score a row whose logits are already aligned by one native forward."""

    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        return _not_measured("identical combined forward did not expose aligned row logits")
    try:
        phases = _segment_token_indices(row_token_ids, contract=contract)
    except OrchestrationError as exc:
        return _not_measured(f"candidate row segment arithmetic failed: {exc}")
    if int(logits.shape[0]) != len(row_token_ids):
        return _not_measured(
            "identical combined forward row logits have wrong token arity",
            expected_tokens=len(row_token_ids),
            observed_tokens=int(logits.shape[0]),
        )
    values = torch.log_softmax(logits.to(dtype=torch.float32), dim=-1)
    selected = torch.stack(
        [values[index, int(token)] for index, token in enumerate(row_token_ids)]
    )
    if not bool(torch.isfinite(selected).all().item()):
        return _not_measured("candidate row log-probabilities are non-finite")
    result: dict[str, Any] = {
        "status": "measured",
        "source": "identical_combined_forward",
        "teacher_forced": True,
        "diagnostic_only": True,
        "segments": {},
    }
    for name, indices in phases.items():
        segment = selected[list(indices)]
        result["segments"][name] = {
            "status": "measured",
            "sum_log_probability": float(segment.sum().item()),
            "mean_log_probability": float(segment.mean().item()),
            "token_count": len(indices),
            "token_indices": list(indices),
            "token_log_probabilities": [float(value) for value in segment.detach().cpu().tolist()],
            "token_ranks": [
                int(1 + (logits[index] > logits[index, int(row_token_ids[index])]).sum().item())
                for index in indices
            ],
        }
    result["selected_token_log_probabilities"] = [float(value) for value in selected.detach().cpu().tolist()]
    return result


def _build_independent_p4_candidate_inputs(
    prefix_ids: Sequence[int],
    candidate_rows: Mapping[str, Sequence[int]],
    *,
    opener_token_id: int,
    device: torch.device | str,
) -> tuple[tuple[int, ...], dict[str, torch.Tensor]]:
    """Build identical-prefix candidate inputs without duplicate row openers."""

    prefix = tuple(int(token) for token in prefix_ids)
    if not prefix or prefix[-1] != int(opener_token_id):
        raise OrchestrationError("P4 exact natural prefix does not end in the native row opener")
    base = prefix[:-1]
    if not base:
        raise OrchestrationError("P4 pre-opener prefix is empty")
    inputs: dict[str, torch.Tensor] = {}
    for role, raw_row in candidate_rows.items():
        row = tuple(int(token) for token in raw_row)
        if not row or row[0] != int(opener_token_id):
            raise OrchestrationError(f"P4 {role} row does not begin with the native opener")
        inputs[str(role)] = torch.tensor(
            [[*base, *row[:-1]]], dtype=torch.long, device=device
        )
    return base, inputs


def _run_gradient_audit(adapter: ExperimentRuntimeAdapter, context: EventContext) -> dict[str, Any]:
    runtime = context.runtime
    prefix = context.prefix_ids
    if not isinstance(adapter.model, torch.nn.Module):
        raise OrchestrationError("P4 requires a live torch.nn.Module HF model")
    if not context.latest_row_ids:
        raise OrchestrationError("P4 requires a latest native complete row")
    regions = _event_regions(context.event)
    if regions is None:
        raise OrchestrationError("P4 requires declared image-cell regions")

    def image_region(key: str) -> tuple[int, ...]:
        values = regions.get(key, ())
        if isinstance(values, (str, bytes)):
            raise OrchestrationError(f"P4 region {key} must be an integer sequence")
        try:
            relative = tuple(sorted({int(value) for value in values}))
        except (TypeError, ValueError):
            raise OrchestrationError(f"P4 region {key} must be an integer sequence") from None
        if not relative:
            raise OrchestrationError(f"P4 region {key} is empty")
        try:
            absolute = tuple(static.resolve_image_positions(runtime.image_span, relative))
        except (TypeError, ValueError) as exc:
            raise OrchestrationError(f"P4 region {key} is outside the image span: {exc}") from exc
        return absolute

    image_positions = image_region("b_exclusive")
    background_positions = image_region("background")
    owner_positions, owner_region_receipts, owner_region_reason = _verified_owner_regions(context, regions)
    owner_state_positions = {
        owner_id: positions
        for owner_id, positions in owner_positions.items()
        if owner_region_receipts.get(owner_id, {}).get("role") in {"non-target-owner", "covered-A"}
    }
    prefix_ids = tuple(int(value) for value in prefix[0].detach().cpu().tolist())
    latest_start = len(prefix_ids) - 1 - len(context.latest_row_ids)
    if latest_start < 0:
        raise OrchestrationError("P4 latest row is not contained in the exact natural prefix")
    terminal_positions = (latest_start + len(context.latest_row_ids) - 1,)
    row_positions = tuple(range(latest_start, latest_start + len(context.latest_row_ids)))
    if set(image_positions) & set(background_positions):
        raise OrchestrationError("P4 image and background spans overlap unexpectedly")
    protected_positions = set(image_positions) | set(background_positions) | set(terminal_positions) | set(row_positions)
    for owner_id, positions_for_owner in owner_positions.items():
        role = owner_region_receipts.get(owner_id, {}).get("role")
        if role == "non-target-owner" and protected_positions & set(positions_for_owner):
            raise OrchestrationError(f"P4 owner region {owner_id} overlaps target/background/history positions")
    if (set(image_positions) | set(background_positions)) & (set(terminal_positions) | set(row_positions)):
        raise OrchestrationError("P4 image state overlaps natural-history row positions")
    if not set(terminal_positions).issubset(set(row_positions)):
        raise OrchestrationError("P4 terminal carrier must be a position in the latest row span")

    # Validate every diagnostic row through the native parser and physical
    # matcher before exposing it to the gradient helper.  A forced row is a
    # diagnostic endpoint only; it is never counted as a natural release.
    for row_name, row_ids in (
        ("target-B", context.target_row_ids),
        ("uncovered-B", context.uncovered_row_ids),
        ("covered-A", context.latest_row_ids),
    ):
        parsed = adapter.parse_row(row_ids, runtime, row_index=0)
        if not parsed.get("valid"):
            raise OrchestrationError(f"P4 {row_name} failed the native parser")

    layer, layer_resolution = dynamic.resolve_block23(adapter.model)
    module_name = next(
        (name for name, module in adapter.model.named_modules() if module is layer),
        None,
    )
    if not module_name:
        raise OrchestrationError("P4 block23 module is not present in named_modules")
    # Score every diagnostic candidate in an independent native forward.  The
    # frozen prefix already ends in the row opener, so each input is the same
    # pre-opener prefix followed by ``row[:-1]``.  This scores the opener at
    # the pre-opener boundary and prevents both duplicate openers and
    # cross-candidate conditioning.
    candidate_rows = {
        "target-B": list(context.target_row_ids),
        "uncovered-B": list(context.uncovered_row_ids),
        "covered-A": list(context.latest_row_ids),
    }
    base_prefix_ids, candidate_inputs = _build_independent_p4_candidate_inputs(
        prefix_ids,
        candidate_rows,
        opener_token_id=adapter.wrapper_contract.object_ref_start_token_id,
        device=adapter.model_device,
    )
    candidate_payloads: dict[str, Mapping[str, Any]] = {}
    candidate_positions: dict[str, torch.Tensor] = {}
    candidate_mrope: dict[str, str] = {}
    for role, ids in candidate_inputs.items():
        payload, positions, mrope_hash = adapter.exact_model_inputs(runtime, ids)
        if payload.get("input_ids") is not ids or payload.get("position_ids") is not positions:
            raise OrchestrationError(f"P4 {role} native payload identity differs from its graph input")
        candidate_payloads[role] = payload
        candidate_positions[role] = positions
        candidate_mrope[role] = str(mrope_hash)
    audit_input_ids = candidate_inputs["target-B"]
    audit_position_ids = candidate_positions["target-B"]
    audit_mrope_hash = candidate_mrope["target-B"]
    history_hash = gradient.sha256_json(
        {
            "checkpoint": adapter.checkpoint,
            "image_id": runtime.image_id,
            "prefix_sha256": sha256_token_ids(prefix_ids),
            "latest_row_sha256": sha256_token_ids(context.latest_row_ids),
        }
    )
    config_payload = (
        adapter.config.model_dump(mode="json")
        if callable(getattr(adapter.config, "model_dump", None))
        else repr(adapter.config)
    )
    config_identity = gradient.sha256_json(config_payload)
    wrapper_identity = gradient.sha256_json(
        {
            "assistant_format": adapter.wrapper_contract.assistant_format,
            "object_ref_start": adapter.wrapper_contract.object_ref_start_token_id,
            "object_ref_end": adapter.wrapper_contract.object_ref_end_token_id,
            "box_start": adapter.wrapper_contract.box_start_token_id,
            "box_end": adapter.wrapper_contract.box_end_token_id,
            "coordinate_start": adapter.wrapper_contract.coordinate_token_start_id,
            "commit": adapter.wrapper_contract.commit_token_id,
        }
    )
    checkpoint_identity = gradient.sha256_json(
        {
            "checkpoint": adapter.checkpoint,
            "h0_root": str(adapter.h0_root),
            "h0_trace_sha256": runtime.h0.get("trace_sha256"),
        }
    )
    runtime_contract = gradient.RuntimeContract(
        wrapper_mode="commit" if adapter.wrapper_contract.commit_token_id is not None else "closed",
        prefix_token_ids=tuple(int(value) for value in audit_input_ids[0].detach().cpu().tolist()),
        expected_prefix_sha256=gradient.sha256_int_sequence(audit_input_ids.reshape(-1)),
        expected_position_ids_sha256=gradient.sha256_position_ids(audit_position_ids),
        object_ref_start_token_id=adapter.wrapper_contract.object_ref_start_token_id,
        object_ref_end_token_id=adapter.wrapper_contract.object_ref_end_token_id,
        box_start_token_id=adapter.wrapper_contract.box_start_token_id,
        box_end_token_id=adapter.wrapper_contract.box_end_token_id,
        coordinate_token_min=adapter.wrapper_contract.coordinate_token_start_id,
        coordinate_token_max=adapter.wrapper_contract.coordinate_token_start_id + adapter.wrapper_contract.coordinate_bin_count - 1,
        coordinate_arity=adapter.wrapper_contract.coordinate_count,
        commit_token_id=adapter.wrapper_contract.commit_token_id,
        block23_module_identity=module_name,
        checkpoint_identity=checkpoint_identity,
        config_identity=config_identity,
        wrapper_identity=wrapper_identity,
        expected_mrope_sha256=audit_mrope_hash,
        expected_image_positions_sha256=gradient.sha256_int_sequence(image_positions),
        expected_background_positions_sha256=gradient.sha256_int_sequence(background_positions),
        expected_terminal_positions_sha256=gradient.sha256_int_sequence(terminal_positions),
        expected_row_span_positions_sha256=gradient.sha256_int_sequence(row_positions),
        expected_natural_history_sha256=history_hash,
        terminal_token_id=adapter.wrapper_contract.closure_token_id,
    )

    def first_block_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
            tensor = value[0]
        elif isinstance(value, Mapping):
            tensor = value.get("last_hidden_state")
        else:
            tensor = None
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 1:
            raise OrchestrationError("P4 block23 hook did not return [1,S,H] hidden states")
        return tensor

    def state_selector(block_outputs: Any) -> Mapping[str, torch.Tensor]:
        if not isinstance(block_outputs, tuple) or len(block_outputs) != len(candidate_rows):
            raise OrchestrationError("P4 expected one block23 output per independent candidate")
        hidden_by_role = {
            role: first_block_tensor(block_output)
            for role, block_output in zip(candidate_rows, block_outputs, strict=True)
        }
        logical_positions: dict[str, tuple[int, ...]] = {
            "image_residual": image_positions,
            "matched_background": background_positions,
            "latest_terminal_carrier": terminal_positions,
            "latest_row_span": row_positions,
            **{f"non_target_owner:{owner_id}": positions for owner_id, positions in owner_state_positions.items()},
        }
        states, _sources = _bind_p4_logical_gradient_sources(
            hidden_by_role,
            logical_positions,
        )
        return states

    def gradient_source_selector(
        block_outputs: tuple[Any, ...],
    ) -> Mapping[str, Sequence[gradient.GradientSource]]:
        if len(block_outputs) != len(candidate_rows):
            raise OrchestrationError("P4 gradient source count differs from candidate count")
        hidden_by_role = {
            role: first_block_tensor(block_output)
            for role, block_output in zip(candidate_rows, block_outputs, strict=True)
        }
        logical_positions: dict[str, tuple[int, ...]] = {
            "image_residual": image_positions,
            "matched_background": background_positions,
            "latest_terminal_carrier": terminal_positions,
            "latest_row_span": row_positions,
            **{f"non_target_owner:{owner_id}": positions for owner_id, positions in owner_state_positions.items()},
        }
        _states, sources = _bind_p4_logical_gradient_sources(
            hidden_by_role,
            logical_positions,
        )
        return sources

    provenance = {
        "image_residual": gradient.StateProvenance(
            role="image_span_b_exclusive", positions=image_positions,
            positions_sha256=gradient.sha256_int_sequence(image_positions),
            span_provenance="b-exclusive-image-span", history_sha256=gradient.sha256_json({"history": "static"}),
            natural_history=False, forward_id="pending", model=adapter.model,
            block23_module=layer, block23_module_name=module_name, input_ids_sha256="pending",
            position_ids_sha256="pending", mrope_sha256="pending",
        ),
        "matched_background": gradient.StateProvenance(
            role="background_control", positions=background_positions,
            positions_sha256=gradient.sha256_int_sequence(background_positions),
            span_provenance="background-control", history_sha256=gradient.sha256_json({"history": "static"}),
            natural_history=False, forward_id="pending", model=adapter.model,
            block23_module=layer, block23_module_name=module_name, input_ids_sha256="pending",
            position_ids_sha256="pending", mrope_sha256="pending",
        ),
        "latest_terminal_carrier": gradient.StateProvenance(
            role="latest_terminal_natural_history", positions=terminal_positions,
            positions_sha256=gradient.sha256_int_sequence(terminal_positions),
            span_provenance="natural-history-terminal", history_sha256=history_hash,
            natural_history=True, forward_id="pending", model=adapter.model,
            block23_module=layer, block23_module_name=module_name, input_ids_sha256="pending",
            position_ids_sha256="pending", mrope_sha256="pending",
        ),
        "latest_row_span": gradient.StateProvenance(
            role="latest_row_span_natural_history", positions=row_positions,
            positions_sha256=gradient.sha256_int_sequence(row_positions),
            span_provenance="natural-history-row-span", history_sha256=history_hash,
            natural_history=True, forward_id="pending", model=adapter.model,
            block23_module=layer, block23_module_name=module_name, input_ids_sha256="pending",
            position_ids_sha256="pending", mrope_sha256="pending",
        ),
    }

    def forward_call() -> Mapping[str, Any]:
        scored: dict[str, torch.Tensor] = {}
        boundary_logits: torch.Tensor | None = None
        for role, row in candidate_rows.items():
            output = adapter.model(**candidate_payloads[role])
            logits = output.logits[0] if hasattr(output, "logits") else output["logits"][0]
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                raise OrchestrationError(f"P4 {role} forward did not expose [sequence,vocab] logits")
            start = len(base_prefix_ids) - 1
            aligned = logits[start : start + len(row)]
            if int(aligned.shape[0]) != len(row):
                raise OrchestrationError(f"P4 {role} candidate logit arity is misaligned")
            scored[role] = aligned
            if role == "target-B":
                boundary_logits = logits[start]
        assert boundary_logits is not None
        return {
            "target_logits": scored["target-B"],
            "uncovered_b_logits": scored["uncovered-B"],
            "covered_a_logits": scored["covered-A"],
            "grammar_logits": scored["target-B"],
            "row_entry_boundary_logits": boundary_logits,
        }

    grammar_token_ids, stop_token_ids, invalid_token_ids, token_mass_contract = _build_token_mass_contract(adapter)
    capture = gradient.capture_native_forward(
        model=adapter.model,
        block23_module=layer,
        block23_module_name=module_name,
        input_ids=audit_input_ids,
        position_ids=audit_position_ids,
        forward_call=forward_call,
        state_selector=state_selector,
        state_provenance=provenance,
        checkpoint_identity=checkpoint_identity,
        config_identity=config_identity,
        wrapper_identity=wrapper_identity,
        mrope_sha256=audit_mrope_hash,
        block23_layer_index=23,
        expected_hook_call_count=3,
        gradient_source_selector=gradient_source_selector,
    )
    batch = gradient.AuditBatch(
        input_ids=audit_input_ids,
        position_ids=audit_position_ids,
        target_row_token_ids=context.target_row_ids,
        uncovered_b_row_token_ids=context.uncovered_row_ids,
        covered_a_row_token_ids=context.latest_row_ids,
        image_residual=capture.captured_states["image_residual"],
        matched_background=capture.captured_states["matched_background"],
        latest_terminal_carrier=capture.captured_states["latest_terminal_carrier"],
        latest_row_span=capture.captured_states["latest_row_span"],
        grammar_token_ids=grammar_token_ids,
        stop_token_ids=stop_token_ids,
        invalid_token_ids=invalid_token_ids,
        non_target_owner_states={
            owner_id: capture.captured_states[f"non_target_owner:{owner_id}"]
            for owner_id in owner_state_positions
            if f"non_target_owner:{owner_id}" in capture.captured_states
        },
        non_target_owner_region_receipts=owner_region_receipts,
    )
    receipt = gradient.run_static_dynamic_gradient_path_audit(
        model=adapter.model,
        runtime=runtime_contract,
        batch=batch,
        forward_fn=lambda _batch: capture,
    )
    receipt["graph_capture"] = {
        "block23_capture_positions": [*image_positions, *background_positions, *terminal_positions, *row_positions],
        "block23_module_name": module_name,
        "block23_layer_index": layer_resolution["layer_idx"],
        "native_forward_id": capture.forward_id,
        "diagnostic_teacher_forced": True,
        "candidate_forward_count": 3,
        "candidate_forward_identity": {
            role: {
                "input_ids_sha256": sha256_token_ids(candidate_inputs[role]),
                "position_ids_sha256": gradient.sha256_position_ids(candidate_positions[role]),
                "mrope_sha256": candidate_mrope[role],
                "base_prefix_sha256": sha256_token_ids(base_prefix_ids),
                "base_prefix_token_count": len(base_prefix_ids),
                "input_token_count": int(candidate_inputs[role].shape[1]),
                "row_token_ids_sha256": sha256_token_ids(candidate_rows[role]),
                "row_appended_without_duplicate_opener": True,
            }
            for role in candidate_rows
        },
        "contract_prefix_input_ids_sha256": sha256_token_ids(prefix_ids),
        "audit_forward_input_ids_sha256": sha256_token_ids(audit_input_ids),
        "audit_forward_position_ids_sha256": gradient.sha256_position_ids(audit_position_ids),
        "candidate_conditioning": "independent_identical_pre_opener_prefix",
        "arbitrary_leaf_injected": False,
    }
    candidate_scores: dict[str, Any] = {}
    for role, key, row in (
        ("target-B", "target_logits", context.target_row_ids),
        ("uncovered-B", "uncovered_b_logits", context.uncovered_row_ids),
        ("covered-A", "covered_a_logits", context.latest_row_ids),
    ):
        logits_value = capture.outputs.get(key) if isinstance(getattr(capture, "outputs", None), Mapping) else None
        candidate_scores[role] = _aligned_row_segment_scores(
            logits_value,
            row,
            contract=adapter.wrapper_contract,
        )
    target_score = candidate_scores.get("target-B")
    covered_score = candidate_scores.get("covered-A")
    score_comparison: dict[str, Any] = {
        "target_B_vs_covered_A": _not_measured("candidate rows have no shared teacher-forced phase score")
        if not isinstance(target_score, Mapping) or target_score.get("status") != "measured" or not isinstance(covered_score, Mapping) or covered_score.get("status") != "measured"
        else {
            "status": "measured",
            "diagnostic_only": True,
            "full_row_mean_logprob_delta": float(
                target_score["segments"]["full_row"]["mean_log_probability"]
                - covered_score["segments"]["full_row"]["mean_log_probability"]
            ),
        },
        "target_B_vs_best_verified_uncovered": _not_measured(
            "no independently verified uncovered candidate bank was bound to this forward"
        ),
    }
    boundary_logits = capture.outputs.get("row_entry_boundary_logits") if isinstance(getattr(capture, "outputs", None), Mapping) else None
    if isinstance(boundary_logits, torch.Tensor) and boundary_logits.ndim == 1 and adapter.wrapper_contract.eos_token_id is not None:
        boundary_log_probs = torch.log_softmax(boundary_logits.to(dtype=torch.float32), dim=-1)
        row_entry_id = adapter.wrapper_contract.object_ref_start_token_id
        terminal_id = adapter.wrapper_contract.eos_token_id
        row_entry_logprob = boundary_log_probs[row_entry_id]
        terminal_logprob = boundary_log_probs[terminal_id]
        if bool(torch.isfinite(torch.stack((row_entry_logprob, terminal_logprob))).all().item()):
            score_comparison["row_entry_vs_native_im_end"] = {
                "status": "measured",
                "source": "identical_combined_forward",
                "teacher_forced": True,
                "diagnostic_only": True,
                "row_entry_token_id": int(row_entry_id),
                "terminal_token_id": int(terminal_id),
                "row_entry_log_probability": float(row_entry_logprob.item()),
                "terminal_log_probability": float(terminal_logprob.item()),
                "row_entry_minus_terminal": float((row_entry_logprob - terminal_logprob).item()),
            }
    else:
        score_comparison["row_entry_vs_native_im_end"] = _not_measured(
            "identical combined forward lacked a finite native row-entry/STOP boundary"
        )
    receipt["candidate_row_scores"] = candidate_scores
    receipt["candidate_score_comparisons"] = score_comparison
    receipt["score_semantics"] = {
        "teacher_forced": True,
        "diagnostic_only": True,
        "natural_release_claim": False,
        "active_intervention_binding": "native_combined_forward_only; interventions must be separately measured",
    }
    receipt["token_mass_contract"] = token_mass_contract
    receipt["owner_region_contract"] = {
        "status": "measured" if owner_region_reason is None else "not_measured",
        "reason": owner_region_reason,
        "regions": owner_region_receipts,
        "shared_core_excluded": True,
        "target_role": "target-B",
        "covered_role": "covered-A",
        "non_target_role": "non-target-owner",
    }
    grouped_owner_effects: dict[str, dict[str, Any]] = {
        "covered-A": {},
        "non-target-owner": {},
    }
    for objective_name, objective_effect in receipt.get("non_target_owner_effects", {}).items():
        if not isinstance(objective_effect, Mapping) or not isinstance(objective_effect.get("owners"), Mapping):
            continue
        for owner_id, owner_effect in objective_effect["owners"].items():
            role = owner_region_receipts.get(str(owner_id), {}).get("role", "non-target-owner")
            grouped_owner_effects.setdefault(str(role), {}).setdefault(objective_name, {})[str(owner_id)] = owner_effect
    receipt["owner_effect_groups"] = grouped_owner_effects
    if owner_region_reason is not None:
        receipt["non_target_owner_effects"] = _not_measured(owner_region_reason)
    return receipt


class OwnerInterfaceOrchestrator:
    """Stage dispatcher and artifact owner for one checkpoint/shard."""

    def __init__(
        self,
        *,
        checkpoint: Literal["S", "A"],
        stage: str,
        output_dir: Path,
        event_shard: tuple[int, int] | None = None,
        event_limit: int | None = None,
        fail_collision: bool = True,
        dry_run: bool = False,
        materialize_ineligible_only: bool = False,
        adapter: Any | None = None,
        config_path: Path | None = None,
        panel_path: Path = DEFAULT_PANEL,
        cohort_path: Path = DEFAULT_COHORT,
        h0_root: Path = DEFAULT_H0_ROOT,
        h0_dir: Path | None = None,
    ) -> None:
        checkpoint = str(checkpoint).upper()
        if checkpoint not in {"S", "A"}:
            raise ValueError("checkpoint must be S or A")
        if stage not in {"p1", "p2", "p3", "p4", "all"}:
            raise ValueError("stage must be p1, p2, p3, p4, or all")
        if event_limit is not None and (isinstance(event_limit, bool) or not isinstance(event_limit, int) or event_limit <= 0):
            raise ValueError("event_limit must be a positive integer")
        if materialize_ineligible_only:
            if stage != "all":
                raise ValueError("materialize-ineligible-only requires stage=all")
            if dry_run:
                raise ValueError("materialize-ineligible-only rejects dry_run")
            if event_limit is not None:
                raise ValueError("materialize-ineligible-only rejects event_limit")
            if event_shard is None:
                raise ValueError("materialize-ineligible-only requires event_shard")
            if event_shard[1] != 4:
                raise ValueError("materialize-ineligible-only requires modulo-4 event_shard")
        self.checkpoint = checkpoint
        self.stage = stage
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.event_shard = event_shard
        self.event_limit = event_limit
        self.fail_collision = bool(fail_collision)
        self.dry_run = bool(dry_run)
        self.materialize_ineligible_only = bool(materialize_ineligible_only)
        self.adapter = adapter
        self.config_path = (config_path or CHECKPOINTS[checkpoint]["config"]).expanduser().resolve()
        self.panel_path = Path(panel_path).expanduser().resolve()
        self.source_panel_path = self.panel_path
        self.cohort_path = Path(cohort_path).expanduser().resolve()
        self.h0_root = Path(h0_root).expanduser().resolve()
        self.h0_dir = None if h0_dir is None else Path(h0_dir).expanduser().resolve()
        self.cohort: dict[str, Any] | None = None
        self._h0_ledger_records: dict[str, dict[str, dict[str, Any]]] = {}
        self._h0_ledger_identity: dict[str, str] = {}
        self._cohort_manifest: dict[str, Any] = {}
        self.events: list[Mapping[str, Any]] = []
        self._event_results: list[dict[str, Any]] = []
        self._gradient_receipts: list[dict[str, Any]] = []
        self._runtime_attestation: dict[str, Any] | None = None
        self._materialization_selection: dict[str, Any] | None = None
        self._artifact_paths = (
            "runtime_identity.json",
            "exact_prefix_manifest.json",
            "intervention_manifest.json",
            "per_event_results.jsonl",
            "gradient_receipt.json",
            "terminal_summary.json",
        )

    def _prepare_output(self) -> None:
        if self.fail_collision and self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"artifact output collision: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_cpu_contract(self) -> dict[str, Any]:
        config_bytes = self.config_path.read_bytes()
        panel_bytes = self.panel_path.read_bytes()
        cohort = _read_json(self.cohort_path)
        if cohort.get("unit_id") != UNIT_ID:
            raise OrchestrationError("cohort unit_id does not match the frozen owner-interface unit")
        if not isinstance(cohort.get("events"), list):
            raise OrchestrationError("cohort events must be a JSON list")
        if any(not isinstance(event, Mapping) for event in cohort["events"]):
            raise OrchestrationError("cohort events must contain only mapping records")
        sources = cohort.get("sources", {})
        if not isinstance(sources, Mapping):
            raise OrchestrationError("cohort sources must be a mapping")
        derived_panel = sources.get("derived_panel")
        source_panel = sources.get("source_panel")
        if not isinstance(derived_panel, Mapping) or not isinstance(source_panel, Mapping):
            raise OrchestrationError("cohort sources must require derived_panel and source_panel mappings")
        cohort_manifest_path = self.cohort_path.with_name(self.cohort_path.name.replace(".json", ".manifest.json"))
        if not cohort_manifest_path.is_file():
            raise OrchestrationError(f"cohort source manifest is missing: {cohort_manifest_path}")
        cohort_manifest = _read_json(cohort_manifest_path)
        if cohort_manifest.get("unit_id") != UNIT_ID:
            raise OrchestrationError("cohort source manifest unit_id does not match the frozen owner-interface unit")
        manifest_sources = cohort_manifest.get("source_hashes")
        if not isinstance(manifest_sources, Mapping):
            raise OrchestrationError("cohort source manifest lacks source_hashes")
        derived_hash = derived_panel.get("sha256")
        source_hash = source_panel.get("sha256")
        if not isinstance(derived_hash, str) or not isinstance(source_hash, str):
            raise OrchestrationError("cohort source_panel and derived_panel require sha256 identities")
        if manifest_sources.get("derived_panel") != derived_hash or manifest_sources.get("source_panel") != source_hash:
            raise OrchestrationError("cohort source panel hashes differ from the cohort manifest")
        derived_path_value = derived_panel.get("path")
        source_path_value = source_panel.get("path")
        if not isinstance(derived_path_value, str) or not isinstance(source_path_value, str):
            raise OrchestrationError("cohort source_panel and derived_panel require paths")
        derived_path = Path(derived_path_value).expanduser().resolve()
        source_path = Path(source_path_value).expanduser().resolve()
        self.source_panel_path = source_path
        if derived_path != self.panel_path:
            raise OrchestrationError("requested panel path differs from cohort derived_panel path")
        if sha256_file(derived_path) != derived_hash:
            raise OrchestrationError("derived panel bytes differ from the cohort/manifest hash")
        if sha256_file(source_path) != source_hash:
            raise OrchestrationError("source panel bytes differ from the cohort/manifest hash")
        # The injected adapter is a CPU-only test seam and may use a synthetic
        # panel, but every real launch must bind the frozen source identities.
        if self.adapter is None and (derived_hash != DERIVED_PANEL_SHA256 or source_hash != PANEL_SOURCE_SHA256):
            raise OrchestrationError("production panel hashes differ from the frozen owner-interface identities")
        self._h0_ledger_records, self._h0_ledger_identity = _load_h0_ledger_records(
            sources=sources,
            manifest_sources=manifest_sources,
            checkpoint=self.checkpoint,
        )
        self._cohort_manifest = cohort_manifest
        self.cohort = cohort
        cohort_events = list(cohort["events"])
        if self.materialize_ineligible_only:
            assert self.event_shard is not None
            events, self._materialization_selection = _partition_ineligible_materialization_shard(
                cohort_events,
                checkpoint=self.checkpoint,
                event_shard=self.event_shard,
            )
        else:
            events = cohort_events
            if self.event_shard is not None:
                shard, count = self.event_shard
                events = [event for index, event in enumerate(events) if index % count == shard]
            if self.event_limit is not None:
                events = events[: int(self.event_limit)]
        self.events = events
        h0_identity: dict[str, Any] | None = None
        if self.adapter is None:
            h0 = _resolve_h0_artifact(self.checkpoint, h0_root=self.h0_root, explicit=self.h0_dir)
            h0_manifest_path = h0 / "run_manifest.json"
            resolved_path = h0 / "configs" / "resolved.json"
            manifest = _read_json(h0_manifest_path)
            resolved_artifact = _read_json(resolved_path) if resolved_path.is_file() else {}
            resolution = resolved_artifact.get("resolution", {}) if isinstance(resolved_artifact, Mapping) else {}
            h0_identity = {
                "root": str(h0),
                "summary_sha256": sha256_file(h0 / "summary.json"),
                "run_manifest_sha256": sha256_file(h0_manifest_path),
                "resolved_config_sha256": sha256_file(resolved_path) if resolved_path.is_file() else None,
                "resolved_config_fingerprint": resolution.get("fingerprint"),
                "manifest_config_fingerprint": manifest.get("resolved_config_fingerprints", {}).get("infer_config"),
                "model_base_path": manifest.get("model_identity", {}).get("base", {}).get("path"),
                "adapter_path": manifest.get("adapter_identity", {}).get("adapter_path"),
                "embedding_delta_path": manifest.get("embedding_delta_identity", {}).get("identity", {}).get("delta_path"),
            }
            if h0_identity["resolved_config_fingerprint"] != h0_identity["manifest_config_fingerprint"]:
                raise OrchestrationError("H0 resolved config fingerprint differs from run_manifest")
        return {
            "schema_version": SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "checkpoint": self.checkpoint,
            "stage": self.stage,
            "execution_mode": (
                "ineligible_contract_materialization"
                if self.materialize_ineligible_only
                else "live_intervention"
            ),
            "config_path": str(self.config_path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "resolved_config_fingerprint": None if h0_identity is None else h0_identity.get("resolved_config_fingerprint"),
            "panel_path": str(self.panel_path),
            "panel_sha256": hashlib.sha256(panel_bytes).hexdigest(),
            "panel_identity": {
                "derived_panel_sha256": derived_hash,
                "source_panel_sha256": source_hash,
                "cohort_manifest_path": str(cohort_manifest_path),
                "cohort_manifest_sha256": sha256_file(cohort_manifest_path),
                "h0_ledger_sha256": dict(self._h0_ledger_identity),
            },
            "cohort_path": str(self.cohort_path),
            "cohort_sha256": sha256_file(self.cohort_path),
            "helper_schema": helper_schema_receipt(),
            "h0": h0_identity,
            "runtime_attestation": {
                "schema_version": RUNTIME_ATTESTATION_SCHEMA_VERSION,
                "status": "pending",
                "passed": False,
            },
            "event_count": len(events),
            "ineligible_materialization": self._materialization_selection,
            "dry_run": self.dry_run,
        }

    def _load_ineligible_materialization_adapter(
        self,
        identity: Mapping[str, Any],
    ) -> ExperimentRuntimeAdapter:
        """Build exact H0 contexts with the processor-only frontend.

        This path deliberately does not import the backend session factory or
        any executable-model loader.  It reconstructs only tokenizer-visible
        prefixes, immutable H0 traces, image-plan geometry, and source-owner
        mappings needed by ``_make_event_context``.
        """

        del identity
        try:
            from src.config.inference import InferConfig, load_infer_config
            from src.config.fingerprint import sha256_json as config_sha256_json
            from src.inference.image_plan import plan_image_batch
            from src.inference.pipeline import _processor_config, _template_config
            from src.inference.prompt import build_prompt_record
            from src.inference.runtime import assemble_frontend
            from src.data import load_raw_examples
        except Exception as exc:
            raise OrchestrationError(
                f"processor-only runtime imports unavailable: {exc}"
            ) from exc

        h0 = _resolve_h0_artifact(
            self.checkpoint,
            h0_root=self.h0_root,
            explicit=self.h0_dir,
        )
        manifest = _read_json(h0 / "run_manifest.json")
        resolved_path = h0 / "configs" / "resolved.json"
        if not resolved_path.is_file():
            raise OrchestrationError(
                f"successful H0 artifact lacks immutable resolved config: {resolved_path}"
            )
        resolved_artifact = _read_json(resolved_path)
        resolved_payload = resolved_artifact.get("config")
        resolution = resolved_artifact.get("resolution")
        if not isinstance(resolved_payload, Mapping) or not isinstance(resolution, Mapping):
            raise OrchestrationError(
                "H0 resolved config artifact has an invalid config/resolution envelope"
            )
        resolved_fingerprint = str(resolution.get("fingerprint", ""))
        manifest_fingerprint = str(
            manifest.get("resolved_config_fingerprints", {}).get("infer_config", "")
        )
        if not resolved_fingerprint or manifest_fingerprint != resolved_fingerprint:
            raise OrchestrationError(
                "H0 resolved config fingerprint does not match run_manifest"
            )
        try:
            config = InferConfig.model_validate(dict(resolved_payload))
        except Exception as exc:
            raise OrchestrationError(
                f"H0 resolved config cannot be materialized: {exc}"
            ) from exc
        try:
            authored = load_infer_config(self.config_path)
            if authored.config.template.assistant_format != config.template.assistant_format:
                raise OrchestrationError("authored config wrapper differs from H0 resolved wrapper")
        except OrchestrationError:
            raise
        except Exception:
            authored = None
        resolved = SimpleNamespace(
            config=config,
            config_dict=dict(resolved_payload),
            fingerprint=resolved_fingerprint,
            entry_config_path=Path(str(resolution.get("entry_config_path", self.config_path))),
        )
        if config.backend.type != "hf":
            raise OrchestrationError("owner-interface processor frontend requires backend.type=hf")
        frontend = assemble_frontend(
            config,
            generation_config_fingerprint=config_sha256_json(
                config.generation.model_dump(mode="json")
            ),
        )
        if getattr(frontend.qwen, "load_model", None) is not False:
            raise OrchestrationError("processor frontend did not attest load_model=False")
        if getattr(frontend.qwen, "model", None) is not None:
            raise OrchestrationError("processor frontend unexpectedly loaded an executable model")
        if manifest.get("terminal_status") != "completed":
            raise OrchestrationError(f"H0 artifact is not completed: {h0}")

        model_identity = manifest.get("model_identity", {})
        adapter_identity = manifest.get("adapter_identity", {})
        embedding_identity = manifest.get("embedding_delta_identity", {})
        if model_identity.get("base", {}).get("path") != config.model.base_model:
            raise OrchestrationError("H0 model identity differs from resolved config")
        if adapter_identity.get("adapter_path") != config.adapter.path:
            raise OrchestrationError("H0 adapter path differs from resolved config")
        delta_path = config.embedding_delta.path
        if embedding_identity.get("identity", {}).get("delta_path") != delta_path:
            raise OrchestrationError("H0 embedding-delta path differs from resolved config")
        adapter_tensor = Path(config.adapter.path) / "adapter_model.safetensors"
        delta_tensor = Path(delta_path) / "special_token_embeddings.safetensors"
        if sha256_file(adapter_tensor) != ADAPTER_SHA256[self.checkpoint]:
            raise OrchestrationError("resolved adapter tensor hash differs from frozen unit")
        if sha256_file(delta_tensor) != EMBEDDING_DELTA_SHA256[self.checkpoint]:
            raise OrchestrationError("resolved embedding-delta tensor hash differs from frozen unit")

        all_raw_values = load_raw_examples(self.panel_path)
        source_raw_values = load_raw_examples(self.source_panel_path)
        raw_values = _select_event_panel_rows(all_raw_values, self.events)
        derived_by_image = {_raw_image_id(raw): raw for raw in all_raw_values}
        source_by_image = {_raw_image_id(raw): raw for raw in source_raw_values}
        if len(derived_by_image) != len(all_raw_values):
            raise OrchestrationError("derived panel contains duplicate image identities")
        if len(source_by_image) != len(source_raw_values):
            raise OrchestrationError("source panel contains duplicate image identities")
        if set(source_by_image) != set(derived_by_image):
            raise OrchestrationError("source and derived panels have different image identities")

        adapter = ExperimentRuntimeAdapter(
            config=config,
            resolved=resolved,
            frontend=frontend,
            session=None,
            model=None,
            tokenizer=frontend.qwen.tokenizer,
            processor=frontend.qwen.processor,
            checkpoint=self.checkpoint,
            h0_root=h0,
            panel_path=self.panel_path,
            panel_rows={},
            h0_rows={},
            h0_ledger_records=self._h0_ledger_records,
        )
        contract = adapter.wrapper_contract
        trace_rows = _trace_rows(h0 / "pred_token_trace.jsonl", contract=contract)
        adapter.h0_rows = _index_trace_rows_by_image(trace_rows, derived_by_image)

        def index_rows_by_image(
            rows: Sequence[Mapping[str, Any]],
            *,
            label: str,
        ) -> dict[str, Mapping[str, Any]]:
            indexed: dict[str, Mapping[str, Any]] = {}
            known = set(derived_by_image)
            for row in rows:
                row_id = row.get("row_id")
                key = str(row_id) if str(row_id) in known else None
                if key is None:
                    match = re.search(r"(\d+)$", str(row_id))
                    if match is not None:
                        candidate = str(int(match.group(1)))
                        if candidate in known:
                            key = candidate
                if key is None:
                    continue
                if key in indexed:
                    raise OrchestrationError(f"H0 {label} rows collide on image {key}")
                indexed[key] = row
            return indexed

        image_plan_rows: list[Mapping[str, Any]] = []
        image_plan_path = h0 / "image_plan.jsonl"
        with image_plan_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise OrchestrationError(
                        f"invalid H0 image-plan JSON at {image_plan_path}:{line_no}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise OrchestrationError(
                        f"H0 image-plan row is not an object at {image_plan_path}:{line_no}"
                    )
                image_plan_rows.append(row)
        h0_image_plans = index_rows_by_image(image_plan_rows, label="image-plan")
        prompt_trace = manifest.get("prompt_trace")
        if not isinstance(prompt_trace, list) or any(
            not isinstance(row, Mapping) for row in prompt_trace
        ):
            raise OrchestrationError("H0 run_manifest prompt_trace must be a list of objects")
        h0_prompt_rows = index_rows_by_image(prompt_trace, label="prompt-trace")

        template = _template_config(config)
        processor = frontend.qwen.processor
        tokenizer = frontend.qwen.tokenizer
        for raw in raw_values:
            image_id = _raw_image_id(raw)
            source_raw = source_by_image.get(image_id)
            if source_raw is None:
                raise OrchestrationError(
                    f"source panel has no image row for derived image {image_id}"
                )
            if image_id not in adapter.h0_rows:
                raise OrchestrationError(f"H0 has no trace row for panel image {image_id}")
            h0_plan = h0_image_plans.get(image_id)
            if h0_plan is None:
                raise OrchestrationError(f"H0 has no image-plan row for panel image {image_id}")
            prompt_receipt = h0_prompt_rows.get(image_id)
            if prompt_receipt is None:
                raise OrchestrationError(f"H0 has no prompt-trace row for panel image {image_id}")

            plan = plan_image_batch(
                [raw],
                components=frontend.qwen,
                processor_config=_processor_config(config),
                row_indices=[0],
            ).rows[0]
            plan_grid = tuple(int(value) for value in plan.expected_image_grid_thw)
            h0_expected_grid = tuple(int(value) for value in h0_plan.get("expected_image_grid_thw", ()))
            h0_observed_grid = tuple(int(value) for value in h0_plan.get("observed_image_grid_thw", ()))
            if (
                h0_plan.get("status") != "ok"
                or len(h0_expected_grid) != 3
                or h0_expected_grid != h0_observed_grid
                or h0_expected_grid != plan_grid
                or h0_plan.get("image_content_sha256") != plan.image_content_sha256
                or h0_plan.get("logical_transform_id") != plan.logical_transform_id
                or int(h0_plan.get("merge_size", -1)) != int(frontend.qwen.processor_identity.merge_size)
            ):
                raise OrchestrationError(f"H0 image-plan identity drift for image {image_id}")
            prompt = build_prompt_record(
                raw,
                template,
                processor=processor,
                row_index=0,
                merged_visual_tokens=plan.merged_visual_tokens,
            )
            prompt_hash = sha256_token_ids(prompt.prompt_token_ids)
            expected_prompt_hash = prompt_receipt.get(
                "backend_executed_prompt_token_ids_sha256"
            )
            if (
                prompt_receipt.get("prompt_token_parity") != "verified"
                or expected_prompt_hash != prompt_receipt.get(
                    "expected_executed_prompt_token_ids_sha256"
                )
                or prompt_hash != expected_prompt_hash
                or len(prompt.prompt_token_ids)
                != prompt_receipt.get("backend_executed_prompt_token_count")
            ):
                raise OrchestrationError(f"H0 prompt identity drift for image {image_id}")
            grid_one = torch.tensor(h0_observed_grid, dtype=torch.long)
            image_token_id = getattr(frontend.qwen.config, "image_token_id", None)
            if image_token_id is None:
                image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            prompt_ids = torch.tensor([prompt.prompt_token_ids], dtype=torch.long)
            span = static.derive_image_span(
                prompt_ids,
                image_token_id=int(image_token_id),
                image_grid_thw=grid_one,
                merge_size=int(frontend.qwen.processor_identity.merge_size),
            )
            owners, owner_mapping, mapping_receipt = _build_source_derived_owner_mapping(
                source_raw,
                raw,
            )
            adapter.panel_rows[image_id] = ImageRuntime(
                image_id=image_id,
                raw=raw,
                prompt_ids=prompt_ids,
                prompt_record=prompt,
                native_inputs={"image_grid_thw": grid_one.reshape(1, 3)},
                image_grid_thw=grid_one,
                image_token_id=int(image_token_id),
                merge_size=int(frontend.qwen.processor_identity.merge_size),
                image_span=span,
                owners=owners,
                h0=adapter.h0_rows[image_id],
                width=int(plan.declared_width),
                height=int(plan.declared_height),
                owner_mapping=owner_mapping,
                mapping_receipt=mapping_receipt,
            )
        adapter.processor_only_identity = {  # type: ignore[attr-defined]
            "load_model": False,
            "model_present": False,
            "backend_session_opened": False,
            "frontend": frontend.qwen.to_artifact_dict(),
            "h0_trace_sha256": sha256_file(h0 / "pred_token_trace.jsonl"),
            "h0_image_plan_sha256": sha256_file(image_plan_path),
        }
        return adapter

    def _load_adapter(self, identity: Mapping[str, Any]) -> Any:
        if self.adapter is not None:
            return self.adapter
        try:
            from src.config.inference import InferConfig, load_infer_config
            from src.config.fingerprint import sha256_json as config_sha256_json
            from src.inference.backend import GenerationPolicy, DecodeRequest, open_backend_session
            from src.inference.image_plan import plan_image_batch
            from src.inference.pipeline import _processor_config, _template_config
            from src.inference.prompt import build_prompt_record
            from src.inference.runtime import assemble_frontend
            from src.data import load_raw_examples
        except Exception as exc:
            raise OrchestrationError(f"real HF runtime imports unavailable: {exc}") from exc
        h0 = _resolve_h0_artifact(self.checkpoint, h0_root=self.h0_root, explicit=self.h0_dir)
        manifest = _read_json(h0 / "run_manifest.json")
        resolved_path = h0 / "configs" / "resolved.json"
        if not resolved_path.is_file():
            raise OrchestrationError(f"successful H0 artifact lacks immutable resolved config: {resolved_path}")
        resolved_artifact = _read_json(resolved_path)
        resolved_payload = resolved_artifact.get("config")
        resolution = resolved_artifact.get("resolution")
        if not isinstance(resolved_payload, Mapping) or not isinstance(resolution, Mapping):
            raise OrchestrationError("H0 resolved config artifact has an invalid config/resolution envelope")
        resolved_fingerprint = str(resolution.get("fingerprint", ""))
        manifest_fingerprint = str(manifest.get("resolved_config_fingerprints", {}).get("infer_config", ""))
        if not resolved_fingerprint or manifest_fingerprint != resolved_fingerprint:
            raise OrchestrationError("H0 resolved config fingerprint does not match run_manifest")
        try:
            config = InferConfig.model_validate(dict(resolved_payload))
        except Exception as exc:
            raise OrchestrationError(f"H0 resolved config cannot be materialized: {exc}") from exc
        # The successful H0 resolution is the live authority.  An authored
        # leaf is retained only as a semantic cross-check because it may have
        # drifted after H0; it is never used to silently replace this config.
        try:
            authored = load_infer_config(self.config_path)
            if authored.config.template.assistant_format != config.template.assistant_format:
                raise OrchestrationError("authored config wrapper differs from H0 resolved wrapper")
        except OrchestrationError:
            raise
        except Exception:
            authored = None
        resolved = SimpleNamespace(
            config=config,
            config_dict=dict(resolved_payload),
            fingerprint=resolved_fingerprint,
            entry_config_path=Path(str(resolution.get("entry_config_path", self.config_path))),
        )
        if config.backend.type != "hf":
            raise OrchestrationError("owner-interface real runtime requires backend.type=hf")
        frontend = assemble_frontend(config, generation_config_fingerprint=config_sha256_json(config.generation.model_dump(mode="json")))
        if manifest.get("terminal_status") != "completed":
            raise OrchestrationError(f"H0 artifact is not completed: {h0}")
        # Bind the H0 runtime identities before loading a GPU model.  These
        # hashes are the exact tensor files named by the frozen unit.
        model_identity = manifest.get("model_identity", {})
        adapter_identity = manifest.get("adapter_identity", {})
        embedding_identity = manifest.get("embedding_delta_identity", {})
        if model_identity.get("base", {}).get("path") != config.model.base_model:
            raise OrchestrationError("H0 model identity differs from resolved config")
        if adapter_identity.get("adapter_path") != config.adapter.path:
            raise OrchestrationError("H0 adapter path differs from resolved config")
        delta_path = config.embedding_delta.path
        if embedding_identity.get("identity", {}).get("delta_path") != delta_path:
            raise OrchestrationError("H0 embedding-delta path differs from resolved config")
        adapter_tensor = Path(config.adapter.path) / "adapter_model.safetensors"
        delta_tensor = Path(delta_path) / "special_token_embeddings.safetensors"
        if sha256_file(adapter_tensor) != ADAPTER_SHA256[self.checkpoint]:
            raise OrchestrationError("resolved adapter tensor hash differs from frozen unit")
        if sha256_file(delta_tensor) != EMBEDDING_DELTA_SHA256[self.checkpoint]:
            raise OrchestrationError("resolved embedding-delta tensor hash differs from frozen unit")
        all_raw_values = load_raw_examples(self.panel_path)
        source_raw_values = load_raw_examples(self.source_panel_path)
        raw_values = _select_event_panel_rows(all_raw_values, self.events)
        derived_by_image = {_raw_image_id(raw): raw for raw in all_raw_values}
        source_by_image = {_raw_image_id(raw): raw for raw in source_raw_values}
        if len(source_by_image) != len(source_raw_values):
            raise OrchestrationError("source panel contains duplicate image identities")
        if set(source_by_image) != set(derived_by_image):
            raise OrchestrationError("source and derived panels have different image identities")
        session_context = open_backend_session(frontend.launch)
        try:
            session = session_context.__enter__()
            from src.inference.hf_backend import HFBackendSession

            if not isinstance(session, HFBackendSession):
                raise OrchestrationError("HF launch opened an unexpected backend session")
            model, tokenizer, processor = session._model, session._tokenizer, session._processor  # noqa: SLF001
            adapter = ExperimentRuntimeAdapter(
                config=config, resolved=resolved, frontend=frontend, session=session, model=model,
                tokenizer=tokenizer, processor=processor, checkpoint=self.checkpoint, h0_root=h0,
                panel_path=self.panel_path, panel_rows={}, h0_rows={},
                h0_ledger_records=self._h0_ledger_records,
            )
            contract = adapter.wrapper_contract
            # Re-parse traces with model-native token IDs and ensure wrapper
            # identity is exact; this catches a stale H0/parser pairing.
            trace_rows = _trace_rows(h0 / "pred_token_trace.jsonl", contract=contract)
            adapter.h0_rows = _index_trace_rows_by_image(trace_rows, derived_by_image)
            template = _template_config(config)
            for raw in raw_values:
                image_id = _raw_image_id(raw)
                source_raw = source_by_image.get(image_id)
                if source_raw is None:
                    raise OrchestrationError(f"source panel has no image row for derived image {image_id}")
                if image_id not in adapter.h0_rows:
                    raise OrchestrationError(f"H0 has no trace row for panel image {image_id}")
                plan = plan_image_batch([raw], components=frontend.qwen, processor_config=_processor_config(config), row_indices=[0]).rows[0]
                prompt = build_prompt_record(raw, template, processor=processor, row_index=0, merged_visual_tokens=plan.merged_visual_tokens)
                prompt_hash = sha256_token_ids(prompt.prompt_token_ids)
                expected_hash = next((item.get("backend_executed_prompt_token_ids_sha256") for item in manifest.get("prompt_trace", []) if item.get("row_id") == image_id or str(item.get("row_id", "")).endswith(image_id)), None)
                if expected_hash is not None and prompt_hash != expected_hash:
                    raise OrchestrationError(f"H0 prompt hash mismatch for image {image_id}")
                request = DecodeRequest(
                    request_id=f"owner-interface:{self.checkpoint}:{image_id}", chat_text=prompt.chat_text,
                    input_prompt_token_ids=tuple(prompt.input_prompt_token_ids), expected_executed_prompt_token_ids=tuple(prompt.prompt_token_ids),
                    image_path=plan.image_path, declared_image_width=plan.declared_width, declared_image_height=plan.declared_height,
                    decoded_image_width=plan.decoded_width, decoded_image_height=plan.decoded_height, image_sha256=plan.image_content_sha256,
                    generation_policy=GenerationPolicy(max_new_tokens=1, repetition_penalty=float(config.generation.repetition_penalty)),
                    expected_image_grid_thw=tuple(plan.expected_image_grid_thw), logical_transform_id=plan.logical_transform_id,
                )
                native_inputs, executed, observed, _media = session._materialize_native_inputs((request,))  # noqa: SLF001
                if tuple(executed[0]) != tuple(prompt.prompt_token_ids):
                    raise OrchestrationError(f"HF processor prompt expansion differs for image {image_id}")
                grid = native_inputs.get("image_grid_thw")
                if not isinstance(grid, torch.Tensor):
                    raise OrchestrationError(f"H0 image grid missing for image {image_id}")
                grid_one = grid.reshape(-1, 3)[0].detach().clone()
                image_token_id = getattr(model.config, "image_token_id", None)
                if image_token_id is None:
                    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
                span = static.derive_image_span(torch.tensor([prompt.prompt_token_ids], dtype=torch.long), image_token_id=int(image_token_id), image_grid_thw=grid_one, merge_size=int(frontend.qwen.processor_identity.merge_size))
                owners, owner_mapping, mapping_receipt = _build_source_derived_owner_mapping(source_raw, raw)
                adapter.panel_rows[image_id] = ImageRuntime(
                    image_id=image_id, raw=raw, prompt_ids=torch.tensor([prompt.prompt_token_ids], dtype=torch.long, device=adapter.model_device),
                    prompt_record=prompt, native_inputs=native_inputs, image_grid_thw=grid_one, image_token_id=int(image_token_id),
                    merge_size=int(frontend.qwen.processor_identity.merge_size), image_span=span, owners=owners, h0=adapter.h0_rows[image_id],
                    width=int(plan.declared_width), height=int(plan.declared_height),
                    owner_mapping=owner_mapping, mapping_receipt=mapping_receipt,
                )
            # Session ownership must outlive the adapter's stage.  Returning the
            # context manager object here is impossible, so retain it and close
            # in ``run`` after the stage.  The context manager's ``__exit__`` is
            # invoked by the caller through ``close`` when available.
            adapter._session_context = session_context  # type: ignore[attr-defined]
            return adapter
        except Exception:
            session_context.__exit__(*sys.exc_info())
            raise

    def run(self) -> dict[str, Any]:
        self._prepare_output()
        identity = self._load_cpu_contract()
        _write_json(self.output_dir / "runtime_identity.json", identity)
        if self.dry_run:
            # Emit the complete artifact surface with zero attempted events so
            # a schema-only launch is inspectable and collision-safe.
            identity = {**identity, "runtime_attestation": _not_applicable_runtime_attestation(reason="dry_run")}
            self._runtime_attestation = identity["runtime_attestation"]
            _write_json(self.output_dir / "runtime_identity.json", identity)
            self._write_outputs(identity, [], gradient_receipts=[])
            terminal = {**identity, "status": "dry_run_schema_validated", "events_attempted": 0}
            _write_json(self.output_dir / "terminal_summary.json", terminal)
            return terminal
        if self.materialize_ineligible_only:
            adapter = self._load_ineligible_materialization_adapter(identity)
            try:
                attestation = _not_applicable_runtime_attestation(
                    reason="ineligible_contract_materialization_cpu_only"
                )
                identity = {
                    **identity,
                    "execution_mode": "ineligible_contract_materialization",
                    "runtime_attestation": attestation,
                    "processor_only_runtime": getattr(
                        adapter,
                        "processor_only_identity",
                        {
                            "load_model": False,
                            "model_present": False,
                            "backend_session_opened": False,
                        },
                    ),
                }
                self._runtime_attestation = attestation
                _write_json(self.output_dir / "runtime_identity.json", identity)
                for event in self.events:
                    if isinstance(adapter, ExperimentRuntimeAdapter):
                        _bind_materialization_event_h0_context(adapter, event)
                    context = _make_event_context(adapter, event)
                    if context.actuator_eligible:
                        raise OrchestrationError(
                            "ineligible materialization selection drifted to an actuator-eligible event"
                        )
                    self._event_results.append(
                        _ineligible_event_result(
                            context,
                            checkpoint=self.checkpoint,
                            runtime_attestation=attestation,
                        )
                    )
                self._write_outputs(
                    identity,
                    self._event_results,
                    gradient_receipts=[],
                )
                return self._terminal_summary(identity, self._event_results)
            finally:
                adapter.close()
        adapter = self._load_adapter(identity)
        try:
            if isinstance(adapter, FakeRuntimeAdapter):
                attestation = _not_applicable_runtime_attestation(reason="cpu_test_adapter")
            else:
                attestation = _attest_live_runtime(adapter)
            identity = {**identity, "runtime_attestation": attestation}
            self._runtime_attestation = attestation
            _write_json(self.output_dir / "runtime_identity.json", identity)
            if isinstance(adapter, FakeRuntimeAdapter) and not adapter.test_only:
                raise OrchestrationError("FakeRuntimeAdapter is reserved for CPU tests")
            if hasattr(adapter, "events") and isinstance(adapter, FakeRuntimeAdapter):
                results = self._run_fake(adapter)
                self._write_outputs(identity, results, gradient_receipts=[])
                return results["terminal_summary"]
            for event in self.events:
                context = _make_event_context(adapter, event)
                result = self._run_event(adapter, context)
                yield_result = result
                self._event_results.append(yield_result)
            self._write_outputs(identity, self._event_results, gradient_receipts=self._gradient_receipts)
            return self._terminal_summary(identity, self._event_results)
        finally:
            try:
                adapter.close()
            finally:
                session_context = getattr(adapter, "_session_context", None)
                if session_context is not None and callable(getattr(session_context, "__exit__", None)):
                    session_context.__exit__(None, None, None)

    def _run_event(self, adapter: ExperimentRuntimeAdapter, context: EventContext) -> dict[str, Any]:
        if not context.actuator_eligible:
            return _ineligible_event_result(
                context,
                checkpoint=self.checkpoint,
                runtime_attestation=self._runtime_attestation,
            )
        regions = _event_regions(context.event)
        prefix_receipt = dict(context.prefix_receipt)
        prefix_receipt.setdefault("model_input", {})
        prefix_receipt["model_input"].update(
            {
                "prefix_sha256": sha256_token_ids(context.prefix_ids),
                "prefix_token_count": int(context.prefix_ids.shape[1]),
            }
        )
        output: dict[str, Any] = {
            "event_id": context.event.get("gt_owner_id"),
            "image_id": context.runtime.image_id,
            "checkpoint": self.checkpoint,
            "runtime_attestation": self._runtime_attestation,
            "eligibility": {
                "status": "eligible",
                "pair_status": getattr(context, "pair_status", None),
                "reason": None,
                "actuators_called": True,
            },
            "prefix": prefix_receipt,
        }
        stages = {self.stage} if self.stage != "all" else {"p1", "p2", "p3", "p4"}
        if "p1" in stages:
            if regions is None:
                output["p1"] = {"status": "invalid/uninterpretable", "reason": "event image-cell regions are absent"}
            else:
                arms: dict[str, Any] = {}
                p1_invalid_reasons: list[str] = []
                for arm in ("K00", "K01", "K10", "K11", "K12", "K13"):
                    try:
                        arms[arm] = _release_row_with_static(adapter, context, arm=arm, regions=regions)
                    except OrchestrationError as exc:
                        arms[arm] = {"status": "invalid/uninterpretable", "reason": str(exc)}
                        p1_invalid_reasons.append(f"{arm}: {exc}")
                noop_checks: dict[str, Any] = {}
                if arms.get("K00", {}).get("status") == "valid" and arms.get("K01", {}).get("status") == "valid":
                    k01_receipt = static.compare_noop_receipts(arms["K00"], arms["K01"], candidate_operator_arms=("K01",))
                    noop_checks["K01"] = k01_receipt
                    if not k01_receipt.get("passed"):
                        p1_invalid_reasons.append("P1 K01 self/no-op receipt is not byte-identical to K00")
                else:
                    p1_invalid_reasons.append("P1 K00/K01 no-op comparison is unavailable")
                for layer in (13, 23):
                    for arm in ("R00", "R10", "R11", "R12"):
                        if arm == "R12" and not regions.get("shared_core", regions.get("shared", ())):
                            arms[f"{arm}_block{layer}"] = {
                                "status": "not_applicable",
                                "reason": "declared shared-core image-cell region is empty",
                            }
                            continue
                        try:
                            arms[f"{arm}_block{layer}"] = _release_row_with_static(adapter, context, arm="K00", regions=regions, residual_layer=layer, residual_arm=arm)
                        except OrchestrationError as exc:
                            arms[f"{arm}_block{layer}"] = {"status": "invalid/uninterpretable", "reason": str(exc)}
                            p1_invalid_reasons.append(f"{arm}_block{layer}: {exc}")
                for arm in ("R00", "R10"):
                    try:
                        arms[f"{arm}_block27"] = _release_row_with_static(
                            adapter, context, arm="K00", regions=regions,
                            residual_layer=27, residual_arm=arm,
                        )
                    except OrchestrationError as exc:
                        arms[f"{arm}_block27"] = {"status": "invalid/uninterpretable", "reason": str(exc)}
                        p1_invalid_reasons.append(f"{arm}_block27: {exc}")
                for layer in (13, 23):
                    candidate = arms.get(f"R00_block{layer}")
                    if isinstance(candidate, Mapping) and candidate.get("status") == "valid":
                        check = static.compare_noop_receipts(
                            arms["K00"],
                            candidate,
                            candidate_operator_arms=("R00",),
                        )
                        if not check.get("passed"):
                            p1_invalid_reasons.append(f"P1 R00 block{layer} self/no-op receipt is invalid")
                        noop_checks[f"R00_block{layer}"] = check
                block27_sentinels: dict[str, Any] = {}
                for arm in ("R00", "R10"):
                    candidate = arms.get(f"{arm}_block27")
                    if arms.get("K00", {}).get("status") == "valid" and isinstance(candidate, Mapping) and candidate.get("status") == "valid":
                        sentinel = static.assess_block27_sentinel(arms["K00"], candidate)
                        block27_sentinels[arm] = sentinel
                        if not sentinel.get("instrumentation_valid"):
                            p1_invalid_reasons.append(f"P1 {arm} block27 sentinel detected behavior or receipt drift")
                    else:
                        p1_invalid_reasons.append(f"P1 {arm} block27 sentinel is unavailable")
                output["p1"] = {
                    "status": "invalid/uninterpretable" if p1_invalid_reasons else "attempted",
                    "arms": arms,
                    "noop_checks": noop_checks,
                    "block27_sentinels": block27_sentinels,
                    "invalid_reasons": p1_invalid_reasons,
                }
        if "p2" in stages:
            arms: dict[str, Any] = {}
            for arm_id in dynamic.DYNAMIC_ARM_IDS:
                try:
                    arms[arm_id] = _release_dynamic_horizon(
                        adapter,
                        context,
                        arm_id=arm_id,
                        horizons=(1, 3),
                    )
                except OrchestrationError as exc:
                    arms[arm_id] = {"status": "invalid/uninterpretable", "reason": str(exc)}
            output["p2"] = {"status": "attempted", "arms": arms}
        if "p3" in stages:
            if regions is None or not context.latest_row_ids:
                output["p3"] = {"status": "invalid/uninterpretable", "reason": "K11 regions or latest row absent"}
            else:
                cells: dict[str, Any] = {}
                for cell_id in dynamic.P3_CELL_IDS:
                    use_static = cell_id in {"Y10", "Y11"}
                    use_dynamic = cell_id in {"Y01", "Y11"}
                    try:
                        cells[cell_id] = _release_dynamic_horizon(
                            adapter,
                            context,
                            arm_id="D10" if use_dynamic else "D00",
                            static_regions=regions if use_static else None,
                            p3_cell_id=cell_id,
                            persistent=True,
                            horizons=(1, 3),
                        )
                        cells[cell_id]["cell_id"] = cell_id
                    except OrchestrationError as exc:
                        cells[cell_id] = {"status": "invalid/uninterpretable", "reason": str(exc), "cell_id": cell_id}
                output["p3"] = {"status": "attempted", "cells": cells}
        if "p4" in stages:
            try:
                receipt = _run_gradient_audit(adapter, context)
                self._gradient_receipts.append({"event_id": context.event.get("gt_owner_id"), "receipt": receipt})
                output["p4"] = receipt
            except OrchestrationError as exc:
                output["p4"] = {"status": "invalid/uninterpretable", "reason": str(exc)}
        return output

    def _run_fake(self, adapter: FakeRuntimeAdapter) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for event in self.events:
            result = adapter.run_event(event, checkpoint=self.checkpoint, stage=self.stage)
            result["runtime_attestation"] = self._runtime_attestation
            results.append(result)
        summary = {"status": "completed", "checkpoint": self.checkpoint, "events_attempted": len(results), "event_results": results}
        return {"events": results, "terminal_summary": summary}

    def _terminal_summary(self, identity: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {**dict(identity), "status": "completed", "events_attempted": len(results), "valid_event_count": sum(item.get("p1", {}).get("status") == "attempted" for item in results)}

    def _write_outputs(self, identity: Mapping[str, Any], payload: Mapping[str, Any], *, gradient_receipts: Sequence[Mapping[str, Any]]) -> None:
        events = payload.get("events", payload) if isinstance(payload, Mapping) else payload
        if isinstance(events, Mapping):
            events = [events]
        events = list(events) if isinstance(events, Sequence) and not isinstance(events, (str, bytes)) else []
        exact = [{"event_id": item.get("event_id"), "image_id": item.get("image_id"), "checkpoint": self.checkpoint, "prefix": item.get("prefix")} for item in events if isinstance(item, Mapping)]
        interventions = [{"event_id": item.get("event_id"), "stages": sorted(key for key in item if key in {"p1", "p2", "p3", "p4"})} for item in events if isinstance(item, Mapping)]
        _write_json(self.output_dir / "exact_prefix_manifest.json", {"schema_version": SCHEMA_VERSION, "identity": identity, "events": exact})
        _write_json(self.output_dir / "intervention_manifest.json", {"schema_version": SCHEMA_VERSION, "helper_schema": helper_schema_receipt(), "events": interventions})
        _write_jsonl(self.output_dir / "per_event_results.jsonl", [item for item in events if isinstance(item, Mapping)])
        _write_json(
            self.output_dir / "gradient_receipt.json",
            {
                "schema_version": gradient.SCHEMA_VERSION,
                "runtime_attestation": identity.get("runtime_attestation"),
                "receipts": list(gradient_receipts),
            },
        )
        terminal = self._terminal_summary(identity, events)
        _write_json(self.output_dir / "terminal_summary.json", terminal)


def _contract_from_config_without_model(config: Any) -> static.WrapperContract:
    """Resolve wrapper IDs from a tokenizer-free config is impossible."""

    # H0 trace validation is repeated once a live tokenizer is loaded.  This
    # sentinel only allows discovery to inspect the trace before model loading;
    # actual use is replaced by the live contract in ``_load_adapter``.
    del config
    return static.WrapperContract(
        assistant_format="object_box_closed",
        object_ref_start_token_id=151646,
        object_ref_end_token_id=151647,
        box_start_token_id=151648,
        box_end_token_id=151649,
        coordinate_token_start_id=151670,
        commit_token_id=None,
        eos_token_id=151645,
    )


def parse_event_shard(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)(?:/|:)(\d+)", str(value).strip())
    if match is None:
        raise ValueError("event-shard must be SHARD/COUNT or SHARD:COUNT")
    shard, count = int(match.group(1)), int(match.group(2))
    if count <= 0 or shard < 0 or shard >= count:
        raise ValueError("event-shard requires 0 <= shard < count and count > 0")
    return shard, count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("p1", "p2", "p3", "p4", "all"), default="all")
    parser.add_argument("--checkpoint", choices=("S", "A"), required=True)
    parser.add_argument("--event-shard", default=None)
    parser.add_argument("--event-limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--h0-root", type=Path, default=DEFAULT_H0_ROOT)
    parser.add_argument("--h0-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--materialize-ineligible-only",
        action="store_true",
        help="CPU-only exact-prefix materialization for conclusively ineligible events",
    )
    parser.add_argument("--fail-collision", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.event_limit is not None and args.event_limit <= 0:
        raise SystemExit("--event-limit must be positive")
    try:
        orchestrator = OwnerInterfaceOrchestrator(
            checkpoint=args.checkpoint,
            stage=args.stage,
            output_dir=args.output_dir,
            event_shard=parse_event_shard(args.event_shard),
            event_limit=args.event_limit,
            fail_collision=args.fail_collision,
            dry_run=args.dry_run,
            materialize_ineligible_only=args.materialize_ineligible_only,
            config_path=args.config,
            panel_path=args.panel,
            cohort_path=args.cohort,
            h0_root=args.h0_root,
            h0_dir=args.h0_dir,
        )
        result = orchestrator.run()
    except (OrchestrationError, FileExistsError, ValueError, OSError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
