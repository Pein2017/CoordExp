#!/usr/bin/env python3
"""Matched M/G x QP/CE Image2299 sequence-optimizer ablation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors.torch import load_file, save_file
from scipy.optimize import minimize
import torch

_SCIPY_MINIMIZE = minimize

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import (  # noqa: E402
    run_image2299_canonical_five_tie_protected_null_sentinel as canonical,
)  # noqa: E402
from scripts.research import run_image2299_dyadic_norm_release_distillation as parent  # noqa: E402
from scripts.research import (  # noqa: E402
    run_image2299_protected_null_output_distillation as protected,
)  # noqa: E402


SCHEMA_VERSION = "image2299.matched_sequence_optimizer_ablation.v2"
UNIT_ID = "2026-08-31-image2299-matched-sequence-optimizer-ablation"
OUTPUT_ROOT = (
    Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration") / UNIT_ID
)
M_RECEIPT = (
    parent.OUTPUT_ROOT
    / "20260830T-image2299-dyadic-norm-release-distillation-v1/receipt.json"
)
M_RECEIPT_SHA256 = "9811bdb632c5c564607537fb80892c38b66a584b31a09da97fa887b6fc90a5e6"
TARGET_PATH, TARGET_SHA256 = canonical.TARGET_PATH, canonical.TARGET_SHA256
M_ROUTE_SHA256 = "c89c6f9900303227cf781caf4a39dc203a09a53b0d319930754cda184529530e"
G_ROUTE_SHA256 = "e238e67122aa46b54cf9290d11093b07a7490746d6730d6f843f8d4ee61d677d"
G_PRE_EOS_SHA256 = "c7aa4f659ff3eca7d81ed6d86046d2f9193530245d186bce37419533f3476b25"
EOS, ROW_TOKENS, MARGIN, NORM_CAP = 151645, 9, 0.01, 9 / 8
OWNER_LEDGER = (
    30,
    23,
    29,
    16,
    7,
    38,
    44,
    31,
    6,
    40,
    28,
    21,
    3,
    24,
    37,
    26,
    13,
    27,
    41,
    25,
    2,
    19,
    39,
    10,
    4,
    17,
    42,
    33,
    5,
    12,
    15,
    36,
    22,
    20,
    1,
    0,
    35,
    32,
    14,
    34,
    18,
)
LRS, MILESTONES = (2**-13, 2**-12, 2**-11, 2**-10), (0, 1, 2, 4, 8, 16, 32, 50)
RESOURCE_BOUND = {
    "gpu_count": 1,
    "world_size": 1,
    "model_load_count_max": 2,
    "wall_time_seconds_max": 3600,
    "max_peak_cuda_reserved_bytes": 16 * 2**30,
    "output_artifact_bytes_max": 200_000_000,
}
MAINLINE_46_ALLOWED = False
MAINLINE_46_PATH = canonical.OUTPUT_ROOT

_hold = parent._hold
_sha256 = parent._sha256
_atomic_json = parent._atomic_json
_tensor_sha256 = parent._tensor_sha256
_capture_route = parent._capture_route
_greedy_route = parent._greedy_route
_evaluate_route = parent._evaluate_route
_surface_snapshot = parent._surface_snapshot
_artifact_bytes = parent._artifact_bytes
_require_one_gpu = parent._require_one_gpu
_row_norms = parent._row_norms
_residual_identity = parent._residual_identity
_full_vocab_violation = parent._full_vocab_violation
_compact_constraints = parent._compact_constraints
base, full_root, recursive, token_ids_sha256 = (
    parent.base,
    parent.full_root,
    parent.recursive,
    parent.token_ids_sha256,
)
MatchedSequenceHold = protected.ProtectedNullHold


def _reject_mainline_input(path: Path) -> None:
    resolved = path.resolve()
    mainline = MAINLINE_46_PATH.resolve()
    if resolved == mainline or mainline in resolved.parents:
        raise _hold("mainline 46/46 input is forbidden")


def _freeze_s_union(
    m_positive_target_ids: Sequence[int], g_positive_target_ids: Sequence[int]
) -> dict[str, Any]:
    ids = sorted({*map(int, m_positive_target_ids), *map(int, g_positive_target_ids)})
    if not ids:
        raise _hold("empty S_union")
    return {
        "ids": ids,
        "count": len(ids),
        "sha256": _sha256_bytes(json.dumps(ids, separators=(",", ":")).encode()),
    }


def _verify_s_union(frozen: Mapping[str, Any], ids: Sequence[int]) -> bool:
    expected = _freeze_s_union(m_positive_target_ids=ids, g_positive_target_ids=())
    if dict(frozen) != expected:
        raise _hold("S_union drift")
    return True


def _build_canonical_route(
    owners: Sequence[str], rows: Sequence[Mapping[str, Any]], *, eos: int
) -> list[int]:
    by_owner = {str(row["owner"]): list(map(int, row["token_ids"])) for row in rows}
    if (
        len(owners) != 41
        or len(set(owners)) != 41
        or any(len(by_owner.get(str(owner), ())) != ROW_TOKENS for owner in owners)
    ):
        raise _hold("static G owner/row construction drift")
    return [token for owner in owners for token in by_owner[str(owner)]] + [int(eos)]


def _static_g_gate(route: Sequence[int], target: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(target.get("rows", ()))
    owners = [str(row.get("owner", "")) for row in rows]
    expected = _build_canonical_route(owners, rows, eos=EOS)
    descriptions = {
        str(row.get("owner", "")): str(row.get("description", "")) for row in rows
    }
    if list(map(int, route)) != expected:
        raise _hold("static G route drift")
    person_count = sum(descriptions[owner] == "person" for owner in owners)
    ties = sorted(owner for owner in owners if descriptions[owner] == "tie")
    if person_count != 38 or ties != ["gt:2299:10", "gt:2299:12", "gt:2299:44"]:
        raise _hold("static G production-matcher owner drift")
    return {
        "passed": True,
        "matched_person_count": person_count,
        "matched_tie_owner_ids": ties,
        "strict_owner_count": len(owners),
        "hard_debt": 0,
        "natural_row_aligned_eos": True,
    }


def _production_static_g_gate(
    tokenizer: Any, binding: Mapping[str, Any]
) -> dict[str, Any]:
    """The actual parser/global matcher admission, before any teacher forcing."""
    evaluation = _evaluate_route(
        tokenizer=tokenizer,
        tokens=binding["g_tokens"],
        contract=binding["runtime_contract"],
        parent_owner_ids=binding["m_owner_ledger"],
        label="matched-g-static-admission",
    )
    generic = recursive._node_gate(
        evaluation, parent_owner_ids=binding["m_owner_ledger"]
    )
    owners = list(map(str, generic.get("matched_owner_ids", ())))
    descriptions = {
        str(row.get("owner", "")): str(row.get("description", ""))
        for row in binding["target"].get("rows", ())
    }
    people = [owner for owner in owners if descriptions.get(owner) == "person"]
    ties = sorted(owner for owner in owners if descriptions.get(owner) == "tie")
    strict = list(
        dict(evaluation.get("ledger", {}))
        .get("matcher", {})
        .get("prediction_receipts", ())
    )
    debt = _same_set_debt(generic)
    if set(owners) != set(binding["m_owner_ledger"]) or len(owners) != 41:
        debt["strict_41_owner_set"] = True
    if len(people) != 38 or ties != ["gt:2299:10", "gt:2299:12", "gt:2299:44"]:
        debt["38_person_3_tie_set"] = True
    if len(strict) != 41 or any(
        item.get("strict_match_status") != "matched" for item in strict
    ):
        debt["strict_matcher_ledger"] = True
    if list(map(int, evaluation.get("generated_token_ids", ()))) != binding["g_tokens"]:
        debt["route_identity"] = True
    return {"passed": not debt, "debt": debt, "evaluation": evaluation}


def _same_set_debt(generic: Mapping[str, Any]) -> dict[str, Any]:
    """Retain real evaluator debt, but remove promotion-only superset debt."""
    debt = dict(generic.get("debt", {}))
    debt.pop("not_proper_superset", None)
    return debt


def _output_delta(
    x: np.ndarray,
    basis: np.ndarray,
    *,
    selected_token_ids: Sequence[int],
    vocab_size: int,
    row_norms: Sequence[float],
) -> np.ndarray:
    rows = np.asarray(x, dtype=np.float64) @ np.asarray(basis, dtype=np.float64).T
    if rows.shape[0] != len(selected_token_ids) or rows.shape[0] != len(row_norms):
        raise _hold("normalized output surface shape drift")
    delta = np.zeros((vocab_size, rows.shape[1]), dtype=np.float64)
    delta[list(map(int, selected_token_ids))] = (
        np.asarray(row_norms, dtype=np.float64)[:, None] * rows
    )
    return delta


def row_balanced_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    row_tokens: int = ROW_TOKENS,
    object_rows: int = 41,
) -> tuple[torch.Tensor, dict[str, int]]:
    if (
        logits.ndim != 2
        or targets.ndim != 1
        or logits.shape[0] != targets.numel()
        or logits.shape[0] != object_rows * row_tokens + 1
    ):
        raise _hold("CE row/EOS action partition drift")
    actions = [
        torch.nn.functional.cross_entropy(
            logits[row * row_tokens : (row + 1) * row_tokens],
            targets[row * row_tokens : (row + 1) * row_tokens],
        )
        for row in range(object_rows)
    ]
    actions.append(torch.nn.functional.cross_entropy(logits[-1:], targets[-1:]))
    return torch.stack(actions).mean(), {
        "object_rows": object_rows,
        "eos_actions": 1,
        "actions": object_rows + 1,
        "tokens": int(targets.numel()),
    }


def _ce_update(
    x: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    hidden: torch.Tensor,
    basis: torch.Tensor,
    lr: float,
    norm_cap: float = NORM_CAP,
) -> dict[str, Any]:
    if x.requires_grad is not True:
        raise _hold("CE coordinates must be trainable")
    adjusted = logits.clone()
    adjusted[:, : x.shape[0]] += (hidden @ basis * x).sum(dim=1, keepdim=True)
    loss = torch.nn.functional.cross_entropy(adjusted, targets)
    optimizer = torch.optim.SGD([x], lr=lr, momentum=0, weight_decay=0)
    optimizer.zero_grad()
    loss.backward()
    gradient_norm = float(x.grad.norm()) if x.grad is not None else 0.0
    optimizer.step()
    with torch.no_grad():
        norm = x.norm()
        if norm > norm_cap:
            x.mul_(norm_cap / norm)
    return {
        "backward": True,
        "gradient_norm": gradient_norm,
        "selected_gradient_rows": list(range(x.shape[0])),
        "competitor_gradient": False,
        "momentum": 0,
        "weight_decay": 0,
        "projected_norm": float(x.norm()),
    }


def _select_milestone(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    passed = [
        candidate for candidate in candidates if candidate.get("warm_passed") is True
    ]
    return (
        min(
            passed,
            key=lambda candidate: (int(candidate["update"]), float(candidate["lr"])),
        )
        if passed
        else None
    )


def _save_warm_payload(output: Path, payload: Mapping[str, Any]) -> Path:
    path = output / "warm_payload.json"
    if path.exists():
        raise _hold(f"refusing overwrite: {path}")
    _atomic_json(path, dict(payload))
    return path


def _load_cold_payload(path: Path, *, expected: Mapping[str, Any]) -> dict[str, Any]:
    actual = _json(path)
    if actual != dict(expected):
        raise _hold("payload schema/surface drift")
    if "prompt_image_checkpoint" in actual and (
        dict(actual.get("reference_release", {})).get("status") != "verified"
        or dict(actual.get("reference_release", {})).get("receipt_sha256") != M_RECEIPT_SHA256
    ):
        raise _hold("payload reference release evidence is missing or drifted")
    return actual


def _finalize_status(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for result in results:
        for identity in ("route", "ledger"):
            warm, cold = (
                result.get(f"warm_{identity}_sha256"),
                result.get(f"cold_{identity}_sha256"),
            )
            if warm is not None and cold is not None and warm != cold:
                return {"status": "HOLD", "payload_promoted": False}
    if any(
        result.get("warm_passed") is True and result.get("cold_passed") is not True
        for result in results
    ):
        return {"status": "HOLD", "payload_promoted": False}
    if any(result.get("outcome") == "technical_hold" for result in results):
        return {"status": "HOLD", "payload_promoted": False}
    if any(result.get("cold_passed") is True for result in results):
        return {"status": "cold_greedy_41_owner_success", "payload_promoted": False}
    if any(result.get("outcome") == "certified_qp_infeasible" for result in results):
        return {"status": "certified_qp_infeasible", "payload_promoted": False}
    if all(
        result.get("recipe_complete") is True or result.get("warm_passed") is False
        for result in results
    ):
        return {"status": "frozen_recipe_negative", "payload_promoted": False}
    return {"status": "HOLD", "payload_promoted": False}


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise _hold(f"unreadable {path}: {error}") from error
    if not isinstance(value, dict):
        raise _hold(f"non-object JSON: {path}")
    return value


def _static_binding() -> dict[str, Any]:
    """CPU-only immutable M/G construction; deliberately performs no runtime load."""
    _reject_mainline_input(M_RECEIPT)
    _reject_mainline_input(TARGET_PATH)
    if not M_RECEIPT.is_file() or _sha256(M_RECEIPT) != M_RECEIPT_SHA256:
        raise _hold("M receipt SHA drift")
    if not TARGET_PATH.is_file() or _sha256(TARGET_PATH) != TARGET_SHA256:
        raise _hold("target library SHA drift")
    receipt, target = _json(M_RECEIPT), _json(TARGET_PATH)
    m = list(map(int, receipt.get("warm_final", {}).get("generated_token_ids", ())))
    predictions = (
        receipt.get("warm_final", {})
        .get("ledger", {})
        .get("matcher", {})
        .get("prediction_receipts", [])
    )
    owners = [str(x.get("strict_match_gt_owner_id", "")) for x in predictions]
    expected = [f"gt:2299:{n}" for n in OWNER_LEDGER]
    rows = {
        str(x.get("owner", "")): list(map(int, x.get("token_ids", ())))
        for x in target.get("rows", [])
    }
    by_owner = {str(x.get("owner", "")): x for x in target.get("rows", [])}
    route_rows = [by_owner[owner] for owner in expected if owner in by_owner]
    g = _build_canonical_route(expected, route_rows, eos=EOS)
    gate = _static_g_gate(g, {"rows": route_rows})
    static_matcher = {
        "person_count": gate["matched_person_count"],
        "tie_owner_ids": gate["matched_tie_owner_ids"],
        "strict_owner_ids": sorted(expected),
        "hard_debt": gate["hard_debt"],
        "final_row_aligned_eos": gate["natural_row_aligned_eos"],
    }
    if (
        receipt.get("status") != "cold_greedy_38_person_success"
        or len(m) != 370
        or token_ids_sha256(m) != M_ROUTE_SHA256
        or owners != expected
        or len(rows) != 46
        or any(len(rows.get(o, [])) != ROW_TOKENS for o in expected)
        or len(g) != 370
        or token_ids_sha256(g[:-1]) != G_PRE_EOS_SHA256
        or token_ids_sha256(g) != G_ROUTE_SHA256
        or static_matcher
        != {
            "person_count": 38,
            "tie_owner_ids": ["gt:2299:10", "gt:2299:12", "gt:2299:44"],
            "strict_owner_ids": sorted(expected),
            "hard_debt": 0,
            "final_row_aligned_eos": True,
        }
    ):
        raise _hold("static M/G parser-global-matcher admission drift")
    return {
        "m_tokens": m,
        "g_tokens": g,
        "m_owner_ledger": expected,
        "static_g": static_matcher,
        "target": target,
    }


def _binding_receipt() -> dict[str, Any]:
    bound = _static_binding()
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "verified",
        "execution_surface": "cpu_only_no_model_load_no_cuda",
        "m_receipt": str(M_RECEIPT),
        "m_receipt_sha256": M_RECEIPT_SHA256,
        "target": str(TARGET_PATH),
        "target_sha256": TARGET_SHA256,
        "m_route_sha256": M_ROUTE_SHA256,
        "g_route_sha256": G_ROUTE_SHA256,
        "g_pre_eos_sha256": G_PRE_EOS_SHA256,
        "m_owner_ledger": bound["m_owner_ledger"],
        "static_g": bound["static_g"],
        "ce": {
            "lrs": list(LRS),
            "updates": 50,
            "milestones": list(MILESTONES),
            "row_actions": 41,
            "eos_actions": 1,
        },
        "margin": MARGIN,
        "norm_cap": NORM_CAP,
        "resource_bound": RESOURCE_BOUND,
        "runner_sha256": _sha256(Path(__file__)),
    }


def _preflight_path(run_id: str) -> Path:
    return OUTPUT_ROOT / base._safe_run_id(run_id) / "preflight.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _stable_sha(value: Any) -> str:
    return _sha256_bytes(json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")).encode())


def _preflight_identity(
    *,
    bound: Mapping[str, Any],
    ordinary_route: Sequence[int],
    positives: Mapping[str, Sequence[Mapping[str, Any]]],
    sequence_surfaces: Mapping[str, Mapping[str, Any]],
    prompts: Any,
    start_surface: Any = None,
    frozen_surface: Any = None,
    sentinels: Any = None,
) -> dict[str, Any]:
    setup = bound["runtime_contract"].get("setup", {})
    prompt_value = (
        prompts.get("prompt", ()) if isinstance(prompts, Mapping)
        else prompts[0] if isinstance(prompts, (list, tuple)) and prompts else ()
    )
    prompt_tokens = list(map(int, np.asarray(prompt_value).reshape(-1).tolist()))
    image_path = getattr(getattr(setup.get("raw_example"), "image", None), "path", None)
    if (
        image_path is None
        or not prompt_tokens
        or token_ids_sha256(prompt_tokens) != parent.PROMPT_TOKEN_SHA256
        or _sha256(Path(image_path)) != parent.IMAGE_SHA256
    ):
        raise _hold("prompt/image identity drift")
    return _json_safe({
            "schema_version": SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "runner_source_sha256": _sha256(Path(__file__)),
            "runner_sha256": _sha256(Path(__file__)),
            "inputs": {
                "m_receipt_sha256": M_RECEIPT_SHA256,
                "target_sha256": TARGET_SHA256,
                "m_route_sha256": M_ROUTE_SHA256,
                "g_route_sha256": G_ROUTE_SHA256,
            },
            "prompt_image_checkpoint": {
                "checkpoint_readback_sha256": _sha256_bytes(json.dumps(
                    _json_safe(bound["runtime_contract"].get("checkpoint_readback")),
                    sort_keys=True, separators=(",", ":")).encode()),
                "prompt_token_ids_sha256": token_ids_sha256(prompt_tokens) if prompt_tokens else None,
                "image_sha256": _sha256(Path(image_path)) if image_path is not None else None,
            },
            "frozen_start_surfaces": {
                "frozen_surface_sha256": parent.FROZEN_SURFACE_SHA256,
                "start_surface_sha256": parent.START_SURFACE_SHA256,
                "live_start_surface_sha256": _stable_sha(start_surface),
                "live_frozen_surface_sha256": _stable_sha(frozen_surface),
                "sentinels_sha256": _stable_sha(sentinels),
            },
            "static_g": bound["static_g"],
            "routes": {
                "ordinary_sha256": token_ids_sha256(ordinary_route),
                "m_sha256": token_ids_sha256(bound["m_tokens"]) if "m_tokens" in bound else M_ROUTE_SHA256,
                "g_sha256": token_ids_sha256(bound["g_tokens"]) if "g_tokens" in bound else G_ROUTE_SHA256,
            },
            "m": _positive_receipt(positives["m"]),
            "g": _positive_receipt(positives["g"]),
            "s_union": list(bound["s_union"]),
            "s_union_count": len(bound["s_union"]),
            "s_union_sha256": bound["s_union_sha256"],
            "sequence_surfaces": sequence_surfaces,
        })


def _require_preflight(run_id: str, expected_sha256: str) -> tuple[dict[str, Any], str]:
    path = _preflight_path(run_id)
    receipt = _json(path)
    sha = _sha256(path)
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("unit_id") != UNIT_ID
        or receipt.get("status") != "preflight_complete"
        or (
            receipt.get("identity", {}).get("runner_source_sha256") is not None
            and receipt["identity"]["runner_source_sha256"] != _sha256(Path(__file__))
        )
        or (
            receipt.get("identity", {}).get("runner_sha256") is not None
            and receipt["identity"]["runner_sha256"] != _sha256(Path(__file__))
        )
    ):
        raise _hold("preflight receipt schema/status drift")
    if sha != expected_sha256:
        raise _hold("preflight receipt SHA drift")
    if not isinstance(receipt.get("identity"), dict):
        raise _hold("preflight identity missing")
    return receipt, sha


def _enforce_resources(*, loads: int, started: float, output: Path | None = None) -> None:
    if (
        loads > RESOURCE_BOUND["model_load_count_max"]
        or time.monotonic() - started > RESOURCE_BOUND["wall_time_seconds_max"]
        or (torch.cuda.is_available()
            and torch.cuda.max_memory_reserved(0) > RESOURCE_BOUND["max_peak_cuda_reserved_bytes"])
        or (output is not None and _artifact_bytes(output) > RESOURCE_BOUND["output_artifact_bytes_max"])
    ):
        raise _hold("execution resource bound exceeded")


def preflight(*, run_id: str) -> Path:
    path = _preflight_path(run_id)
    if path.exists() or path.parent.exists():
        raise _hold(f"refusing overwrite: {path.parent}")
    path.parent.mkdir(parents=True)
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "run_id": path.parent.name,
        "status": "technical_hold",
        "counts": {"model_loads": 0},
    }
    try:
        _require_one_gpu()
        bound = _static_binding()
        bound["runtime_contract"] = parent._binding_contract()["runtime_contract"]
        setup = bound["runtime_contract"]["setup"]
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        with base.open_backend_session(setup["frontend"].launch) as opened:
            if (
                type(opened) is not base.HFBackendSession
                or opened._model is None
                or opened._tokenizer is None
            ):
                raise _hold("preflight requires concrete FP32 HF backend")
            receipt["counts"]["model_loads"] = 1
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            sentinels = full_root._full_root_nontrainable_sentinels(model)
            static_g = _production_static_g_gate(tokenizer, bound)
            if not static_g["passed"]:
                raise _hold("preflight static G parser/global matcher drift")
            native, prompts, _grids, _media = opened._materialize_native_inputs(
                setup["requests"][:1]
            )
            head = model.get_output_embeddings()
            names, parameters, before, frozen_before = _surface_snapshot(model)
            ordinary_route = _greedy_route(model, native, int(tokenizer.pad_token_id))
            ordinary = _capture_route(
                model=model,
                output_head=head,
                native_inputs=native,
                route_tokens=ordinary_route,
                pad_token_id=int(tokenizer.pad_token_id),
            )
            captures = {
                name: _capture_route(
                    model=model,
                    output_head=head,
                    native_inputs=native,
                    route_tokens=route,
                    pad_token_id=int(tokenizer.pad_token_id),
                )
                for name, route in (("m", bound["m_tokens"]), ("g", bound["g_tokens"]))
            }
            positives = {
                name: _states(captures[name], bound[f"{name}_tokens"])
                for name in ("m", "g")
            }
            frozen = _freeze_s_union(
                [state["target_token_id"] for state in positives["m"]],
                [state["target_token_id"] for state in positives["g"]],
            )
            bound.update({"s_union": frozen["ids"], "s_union_sha256": frozen["sha256"]})
            surfaces = {}
            for name in ("m", "g"):
                protected_hidden, basis, info = _basis(
                    ordinary_route,
                    ordinary,
                    bound[f"{name}_tokens"],
                    captures[name],
                    positives[name],
                )
                surfaces[name] = {
                    "positive_count": len(positives[name]),
                    "positive_rank": int(basis.shape[1]),
                    "protected_count": len(protected_hidden),
                    "protected_sha256": info["protected_hidden_sha256"],
                    "basis_sha256": info["basis_sha256"],
                    "s_union_sha256": frozen["sha256"],
                    "s_union_count": frozen["count"],
                }
            names_after, parameters_after, after, frozen_after = _surface_snapshot(
                model
            )
            if (
                names != names_after
                or any(
                    a is not b
                    for a, b in zip(parameters, parameters_after, strict=True)
                )
                or before != after
                or frozen_before != frozen_after
            ):
                raise _hold("preflight base surface mutation")
            full_root._assert_full_root_sentinels(model, sentinels)
            identity = _preflight_identity(
                bound=bound,
                ordinary_route=ordinary_route,
                positives=positives,
                sequence_surfaces=surfaces,
                prompts=prompts,
                start_surface=before,
                frozen_surface=frozen_before,
                sentinels=sentinels,
            )
            receipt.update(
                {
                    "status": "preflight_complete",
                    "binding": _binding_receipt(),
                    "identity": identity,
                    "runtime": opened.receipt.to_artifact_dict(),
                }
            )
    except BaseException as error:
        receipt["stop"] = {"reason": str(error), "error_type": type(error).__name__}
    finally:
        try:
            _enforce_resources(loads=receipt["counts"]["model_loads"], started=started)
        except BaseException as error:
            receipt.update(status="technical_hold", stop={"reason": str(error), "error_type": type(error).__name__})
        receipt["wall_time_seconds"] = time.monotonic() - started
        receipt["resources"] = {
            "model_loads": receipt["counts"]["model_loads"],
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0))
            if torch.cuda.is_available()
            else 0,
            "predeclared_bound": RESOURCE_BOUND,
        }
        # SHA is intentionally omitted: self-hashing a JSON artifact is not stable.
        _atomic_json(path, _json_safe(receipt))
    if receipt["status"] != "preflight_complete":
        raise _hold(receipt.get("stop", {}).get("reason", "preflight failed"))
    return path


def _prepare(output: Path) -> dict[str, str]:
    if output.exists():
        raise _hold(f"refusing overwrite: {output}")
    output.mkdir(parents=True)
    snapshot = output / "runner_source.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    if snapshot.read_bytes() != Path(__file__).read_bytes():
        raise _hold("successor-local snapshot readback drift")
    return {"path": str(snapshot), "sha256": _sha256(snapshot)}


def _states(
    capture: Mapping[str, torch.Tensor], tokens: Sequence[int]
) -> list[dict[str, Any]]:
    top = capture["logits"].argmax(dim=1).tolist()
    return [
        {
            "position": i,
            "target_token_id": int(t),
            "hidden": capture["hidden"][i].double().numpy(),
            "logits": capture["logits"][i],
        }
        for i, (t, p) in enumerate(zip(tokens, top, strict=True))
        if int(t) != int(p)
    ]


def _all_states(
    capture: Mapping[str, torch.Tensor], tokens: Sequence[int]
) -> list[dict[str, Any]]:
    return [
        {
            "position": i,
            "target_token_id": int(token),
            "hidden": capture["hidden"][i].double().numpy(),
            "logits": capture["logits"][i],
        }
        for i, token in enumerate(tokens)
    ]


def _basis(
    ordinary_route: Sequence[int],
    ordinary: Mapping[str, torch.Tensor],
    route: Sequence[int],
    capture: Mapping[str, torch.Tensor],
    states: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    protected_hidden, selection = protected._protected_hidden_matrix(
        ordinary_route=ordinary_route,
        ordinary_hidden=ordinary["hidden"],
        controlled_route=route,
        controlled_hidden=capture["hidden"],
        controlled_positive_positions=[s["position"] for s in states],
    )
    positive_hidden = np.stack([s["hidden"] for s in states])
    q0, protected_info = protected._row_basis(protected_hidden)
    projected = positive_hidden - (positive_hidden @ q0) @ q0.T
    original_norms = np.linalg.norm(positive_hidden, axis=1)
    projected_norms = np.linalg.norm(projected, axis=1)
    if (
        len(states) == 0
        or np.any(original_norms < 1e-3)
        or np.any(projected_norms < 1e-3)
    ):
        raise _hold("protected-null positive state violates the frozen norm floor")
    b, projected_info = protected._row_basis(projected)
    # SVD projection can leave ~1e-9 cancellation residue on the wider G basis.
    # Reproject and QR a fixed three times; this changes only numerical
    # orthogonalization, not the protected-null span or optimizer contrast.
    for _ in range(3):
        b = b - q0 @ (q0.T @ b)
        b, _r = np.linalg.qr(b, mode="reduced")
    null_max = float(np.max(np.abs(protected_hidden @ b), initial=0.0))
    if not 1 <= b.shape[1] <= len(states) or null_max > 1e-10:
        raise _hold("protected-null positive basis is rank-invalid or numerically non-null")
    info = {
        "protected": protected_info,
        "projected_positive": projected_info,
        "positive_count": len(states),
        "rank": int(b.shape[1]),
        "original_norm_min": float(original_norms.min()),
        "projected_norm_min": float(projected_norms.min()),
        "protected_null_max_abs": null_max,
        "basis_sha256": _tensor_sha256(b),
        "reorthogonalization_passes": 3,
    }
    info["selection"] = selection
    info["protected_hidden_sha256"] = _tensor_sha256(protected_hidden)
    return protected_hidden, b, info


def _constraints(
    states: Sequence[Mapping[str, Any]], basis: np.ndarray, selected: Sequence[int]
) -> list[dict[str, Any]]:
    selected = list(map(int, selected))
    selected_set = set(selected)
    out = []
    for state in states:
        logits, target = state["logits"], int(state["target_token_id"])
        fixed = logits.clone()
        fixed[torch.tensor(selected)] = -torch.inf
        competitor = int(fixed.argmax())
        for c in [*sorted(selected_set - {target}), competitor]:
            out.append(
                {
                    "position": state["position"],
                    "target_token_id": target,
                    "competitor_token_id": c,
                    "competitor_trainable": c in selected_set,
                    "kind": "movable_target_partition"
                    if c in selected_set
                    else "fixed_non_target_partition_max",
                    "raw_margin": float(logits[target] - logits[c]),
                    "feature": state["hidden"] @ basis,
                }
            )
    return out


def _qp(
    states: Sequence[Mapping[str, Any]],
    basis: np.ndarray,
    norms: Mapping[int, float],
    selected: Sequence[int],
) -> dict[str, Any]:
    constraints = _constraints(states, basis, selected)
    # Keep the solver implementation single-sourced in the repaired parent.
    # The narrow hook preserves the established unit-test fault injection seam.
    module = sys.modules[parent._solve_target_only_minimum_normalized.__module__]
    original_minimize = module.minimize
    if minimize is not _SCIPY_MINIMIZE:
        def _fault_injected_minimize(*args: Any, **kwargs: Any) -> Any:
            outcome = minimize(*args, **kwargs)
            if not hasattr(outcome, "nit"):
                outcome.nit = 0
            if not hasattr(outcome, "fun"):
                outcome.fun = math.inf
            return outcome
        module.minimize = _fault_injected_minimize
    try:
        solved = parent._solve_target_only_minimum_normalized(
            constraints, rank=basis.shape[1], row_norms=norms, target_token_ids=selected
        )
    finally:
        module.minimize = original_minimize
    classification = str(solved.get("solver_classification", "numerical_solver_hold"))
    result = {
        "feasible": bool(solved.get("feasible")),
        "classification": (
            "certified_infeasible" if classification == "certified_infeasible"
            else "feasible" if solved.get("feasible") else "HOLD"
        ),
        "constraints": _json_safe(_compact_constraints(constraints)),
        "solver": _json_safe({
            key: value for key, value in solved.items()
            if key not in {"normalized_rows", "scaled_basis_rows"}
        }),
        # Compatibility aliases; diagnostics remain typed and JSON-safe above.
        "highs": _json_safe(solved.get("lp_diagnostics")),
        "slsqp": _json_safe(solved.get("primal_diagnostics")),
    }
    if result["feasible"]:
        rows = np.asarray(solved["scaled_basis_rows"], dtype=np.float64) @ basis.T
        result.update({
            "rows": rows.tolist(),
            "normalized_norm": float(solved["normalized_norm"]),
            "minimum_slack": float(solved["minimum_slack"]),
        })
    return result


def _project(x: torch.Tensor) -> None:
    with torch.no_grad():
        norm = x.norm()
        if norm > NORM_CAP:
            x.mul_(NORM_CAP / norm)


def _ce(
    all_states: Sequence[Mapping[str, Any]],
    positives: Sequence[Mapping[str, Any]],
    basis: np.ndarray,
    norms: Mapping[int, float],
    selected: Sequence[int],
    *,
    warm_evaluate: Any = None,
) -> dict[str, Any]:
    """True target-token CE on the shared normalized output-only surface."""
    selected = list(map(int, selected))
    if len(all_states) != 41 * ROW_TOKENS + 1:
        raise _hold("CE route must be exactly 41 rows plus EOS")
    ids = torch.tensor(selected, dtype=torch.long)
    norm = torch.tensor([norms[token] for token in selected], dtype=torch.float64)
    basis_t = torch.from_numpy(np.asarray(basis, dtype=np.float64))
    traces = []
    for lr in LRS:
        x = torch.zeros(
            (len(selected), basis.shape[1]), dtype=torch.float64, requires_grad=True
        )
        optimizer = torch.optim.SGD([x], lr=lr, momentum=0, weight_decay=0)
        milestones: dict[int, dict[str, Any]] = {}
        for step in range(51):
            if step in MILESTONES:
                rows = (x.detach().numpy() * norm.numpy()[:, None]) @ basis.T
                candidate = _candidate(positives, selected, rows)
                candidate["normalized_norm"] = float(x.detach().norm())
                warm = (
                    warm_evaluate(np.asarray(candidate["rows"], dtype=np.float64))
                    if warm_evaluate
                    else None
                )
                candidate["warm"] = warm
                candidate["warm_passed"] = bool(warm and warm.get("passed") is True)
                milestones[step] = candidate
            if step == 50:
                break
            action_losses = []
            for row in range(41):
                row_losses = []
                for state in all_states[row * ROW_TOKENS : (row + 1) * ROW_TOKENS]:
                    logits = state["logits"].double().clone()
                    feature = (
                        torch.from_numpy(np.asarray(state["hidden"], dtype=np.float64))
                        @ basis_t
                    )
                    logits[ids] += (x * (norm[:, None] * feature[None, :])).sum(dim=1)
                    row_losses.append(
                        torch.nn.functional.cross_entropy(
                            logits.unsqueeze(0),
                            torch.tensor([state["target_token_id"]]),
                        )
                    )
                action_losses.append(torch.stack(row_losses).mean())
            eos_state = all_states[-1]
            eos_logits = eos_state["logits"].double().clone()
            eos_feature = (
                torch.from_numpy(np.asarray(eos_state["hidden"], dtype=np.float64))
                @ basis_t
            )
            eos_logits[ids] += (x * (norm[:, None] * eos_feature[None, :])).sum(dim=1)
            action_losses.append(
                torch.nn.functional.cross_entropy(
                    eos_logits.unsqueeze(0),
                    torch.tensor([eos_state["target_token_id"]]),
                )
            )
            if len(action_losses) != 42:
                raise _hold("CE row/EOS action partition drift")
            optimizer.zero_grad()
            torch.stack(action_losses).mean().backward()
            optimizer.step()
            _project(x)
        traces.append(
            {
                "lr": lr,
                "momentum": 0,
                "weight_decay": 0,
                "updates": 50,
                "milestones": milestones,
            }
        )
    choices = [
        {
            "update": step,
            "lr": trace["lr"],
            "warm_passed": trace["milestones"][step]["warm_passed"],
            "candidate": trace["milestones"][step],
        }
        for trace in traces
        for step in MILESTONES
    ]
    selected_milestone = _select_milestone(choices)
    return {
        "traces": traces,
        "selected": None
        if selected_milestone is None
        else (
            selected_milestone["update"],
            selected_milestone["lr"],
            selected_milestone["candidate"],
        ),
    }


def _candidate(
    states: Sequence[Mapping[str, Any]], selected: Sequence[int], rows: np.ndarray
) -> dict[str, Any]:
    violation = _full_vocab_violation(
        states, selected_token_ids=selected, residual_rows=rows
    )
    return {
        "passed": violation is None,
        "rows": np.asarray(rows, dtype=np.float64).tolist(),
        "full_vocab": None
        if violation is None
        else {
            "position": violation["state"]["position"],
            "competitor": violation["competitor_token_id"],
            "margin": violation["corrected_margin"],
        },
        "normalized_norm": None,
    }


def _save_payload(
    output: Path,
    cell: str,
    selected: Sequence[int],
    rows: np.ndarray,
    surface: Mapping[str, Any],
) -> dict[str, Any]:
    path = output / f"{cell}_output_residual.safetensors"
    save_file(
        {
            "selected_token_ids": torch.tensor(selected, dtype=torch.int64),
            "residual_rows": torch.from_numpy(rows),
        },
        path,
        metadata={"schema_version": SCHEMA_VERSION, "unit_id": UNIT_ID, "cell": cell},
    )
    identity = _residual_identity(selected, rows) | {"payload_sha256": _sha256(path)}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "cell": cell,
        "identity": identity,
        "surface": surface,
    }
    _atomic_json(path.with_suffix(".json"), metadata)
    return {
        "payload": str(path),
        "metadata": str(path.with_suffix(".json")),
        "identity": identity,
    }


def _warm_gate(
    model: Any,
    tokenizer: Any,
    native: Mapping[str, Any],
    head: Any,
    selected: Sequence[int],
    rows: np.ndarray,
    binding: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """The payload admission gate is ordinary greedy, never teacher forcing."""
    wrapper = parent.SparseOutputResidual(head, selected, torch.from_numpy(rows)).to(
        next(model.parameters()).device
    )
    model.set_output_embeddings(wrapper)
    try:
        route = _greedy_route(model, native, int(tokenizer.pad_token_id))
        evaluation = _evaluate_route(
            tokenizer=tokenizer,
            tokens=route,
            contract=binding["runtime_contract"],
            parent_owner_ids=binding["m_owner_ledger"],
            label=label,
        )
    finally:
        model.set_output_embeddings(head)
    generic = recursive._node_gate(
        evaluation, parent_owner_ids=binding["m_owner_ledger"]
    )
    owners = list(map(str, generic.get("matched_owner_ids", ())))
    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    debt = _same_set_debt(generic)
    if set(owners) != set(binding["m_owner_ledger"]) or len(owners) != 41:
        debt["strict_41_owner_set"] = True
    if len(tokens) != 370 or tokens[-1:] != [EOS] or EOS in tokens[:-1]:
        debt["natural_row_aligned_eos"] = True
    return {
        "passed": not debt,
        "debt": debt,
        "generated_token_ids": tokens,
        "generated_token_ids_sha256": token_ids_sha256(tokens),
        "ledger": evaluation["ledger"],
        "ledger_sha256": _sha256_bytes(
            json.dumps(
                _json_safe(evaluation["ledger"]), sort_keys=True, separators=(",", ":")
            ).encode()
        ),
    }


def _cold_verify(*, payload: Path, result: Path, cell: str) -> None:
    """Fresh-process verifier; it never imports a warm model or mutable receipt."""
    if result.exists():
        raise _hold(f"refusing overwrite: {result}")
    _require_one_gpu()
    sequence, optimizer_kind = cell.split("-", 1)
    metadata = _json(payload.with_suffix(".json"))
    tensors = load_file(payload, device="cpu")
    selected_tensor, rows_tensor = (
        tensors.get("selected_token_ids"),
        tensors.get("residual_rows"),
    )
    if (
        selected_tensor is None
        or rows_tensor is None
        or rows_tensor.dtype != torch.float64
        or not bool(torch.isfinite(rows_tensor).all())
    ):
        raise _hold("cold payload tensor schema is invalid")
    selected, rows = list(map(int, selected_tensor.tolist())), rows_tensor.numpy()
    identity = _residual_identity(selected, rows) | {"payload_sha256": _sha256(payload)}
    surface = dict(metadata.get("surface", {}))
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("unit_id") != UNIT_ID
        or metadata.get("cell") != f"{sequence}_{optimizer_kind}"
        or metadata.get("identity") != identity
    ):
        raise _hold("cold payload schema/surface is invalid")
    preflight_ref = surface.get("preflight")
    if not isinstance(preflight_ref, Mapping):
        raise _hold("cold payload preflight identity is missing")
    preflight_receipt, preflight_sha256 = _require_preflight(
        str(preflight_ref.get("run_id", "")), str(preflight_ref.get("sha256", ""))
    )
    if dict(preflight_ref.get("identity", {})) != preflight_receipt["identity"]:
        raise _hold("cold preflight identity drift")
    bound = _static_binding()
    bound["runtime_contract"] = parent._binding_contract()["runtime_contract"]
    setup = bound["runtime_contract"]["setup"]
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    with base.open_backend_session(setup["frontend"].launch) as opened:
        if (
            type(opened) is not base.HFBackendSession
            or opened._model is None
            or opened._tokenizer is None
        ):
            raise _hold("cold verification requires concrete FP32 HF backend")
        model, tokenizer = opened._model, opened._tokenizer
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        sentinels = full_root._full_root_nontrainable_sentinels(model)
        static_g = _production_static_g_gate(tokenizer, bound)
        if not static_g["passed"]:
            raise _hold("cold static G parser/global matcher drift")
        native, prompts, _grids, _media = opened._materialize_native_inputs(
            setup["requests"][:1]
        )
        names, parameters, before, frozen_before = _surface_snapshot(model)
        head = model.get_output_embeddings()
        ordinary_route = _greedy_route(model, native, int(tokenizer.pad_token_id))
        ordinary = _capture_route(
            model=model,
            output_head=head,
            native_inputs=native,
            route_tokens=ordinary_route,
            pad_token_id=int(tokenizer.pad_token_id),
        )
        captures = {
            name: _capture_route(
                model=model,
                output_head=head,
                native_inputs=native,
                route_tokens=route,
                pad_token_id=int(tokenizer.pad_token_id),
            )
            for name, route in (("m", bound["m_tokens"]), ("g", bound["g_tokens"]))
        }
        positives = {
            name: _states(captures[name], bound[f"{name}_tokens"])
            for name in ("m", "g")
        }
        frozen_union = _freeze_s_union(
            m_positive_target_ids=[
                state["target_token_id"] for state in positives["m"]
            ],
            g_positive_target_ids=[
                state["target_token_id"] for state in positives["g"]
            ],
        )
        if (
            surface.get("s_union") != frozen_union["ids"]
            or surface.get("s_union_sha256") != frozen_union["sha256"]
            or selected != frozen_union["ids"]
        ):
            raise _hold("cold S_union/selected-row drift")
        bound.update(
            {"s_union": frozen_union["ids"], "s_union_sha256": frozen_union["sha256"]}
        )
        sequence_surfaces = {}
        for name in ("m", "g"):
            protected_matrix, sequence_basis, sequence_info = _basis(
                ordinary_route,
                ordinary,
                bound[f"{name}_tokens"],
                captures[name],
                positives[name],
            )
            sequence_surfaces[name] = {
                "positive_count": len(positives[name]),
                "positive_rank": int(sequence_basis.shape[1]),
                "protected_count": len(protected_matrix),
                "protected_sha256": sequence_info["protected_hidden_sha256"],
                "basis_sha256": sequence_info["basis_sha256"],
                "s_union_sha256": frozen_union["sha256"],
                "s_union_count": frozen_union["count"],
            }
        if (
            _preflight_identity(
                bound=bound,
                ordinary_route=ordinary_route,
                positives=positives,
                sequence_surfaces=sequence_surfaces,
                prompts=prompts,
                start_surface=before,
                frozen_surface=frozen_before,
                sentinels=sentinels,
            )
            != preflight_receipt["identity"]
        ):
            raise _hold("cold preflight identity recapture drift")
        route, capture, states = (
            bound[f"{sequence}_tokens"],
            captures[sequence],
            positives[sequence],
        )
        protected_hidden, basis, info = _basis(
            ordinary_route, ordinary, route, capture, states
        )
        if (
            surface.get("basis_sha256") != info["basis_sha256"]
            or surface.get("protected_sha256") != info["protected_hidden_sha256"]
            or tuple(rows.shape) != (len(selected), basis.shape[0])
        ):
            raise _hold("cold basis/protected/payload surface drift")
        null = float(np.max(np.abs(protected_hidden @ rows.T), initial=0.0))
        if null > 1e-10:
            raise _hold("cold protected-null violation")
        warm = _warm_gate(
            model,
            tokenizer,
            native,
            head,
            selected,
            rows,
            bound,
            f"matched-{sequence}-{optimizer_kind}-cold",
        )
        names_after, parameters_after, after, frozen_after = _surface_snapshot(model)
        if (
            names != names_after
            or any(
                left is not right
                for left, right in zip(parameters, parameters_after, strict=True)
            )
            or before != after
            or frozen_before != frozen_after
        ):
            raise _hold("cold base surface mutation")
        full_root._assert_full_root_sentinels(model, sentinels)
        _atomic_json(
            result,
            {
                "schema_version": SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "status": "cold_verification_complete",
                "cell": cell,
                "cold_passed": warm["passed"],
                "gate": warm,
                "identity": identity,
                "surface": surface,
                "preflight_sha256": preflight_sha256,
                "basis_sha256": info["basis_sha256"],
                "protected_hidden_sha256": info["protected_hidden_sha256"],
                "protected_null_max_abs": null,
                "runtime": opened.receipt.to_artifact_dict(),
                "resources": {
                    "model_loads": 1,
                    "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                    "predeclared_bound": RESOURCE_BOUND,
                },
            },
        )


def _run_cold(
    snapshot: Path, payload: Mapping[str, Any], output: Path, cell: str, started: float
) -> dict[str, Any]:
    result = output / "cold_verification.json"
    environment = dict(os.environ)
    repo = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = repo + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(snapshot),
            "--cold-verify",
            "--payload",
            str(payload["payload"]),
            "--cold-result",
            str(result),
            "--cell",
            cell,
        ],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        timeout=max(
            1.0, RESOURCE_BOUND["wall_time_seconds_max"] - (time.monotonic() - started)
        ),
        check=False,
    )
    if completed.returncode != 0 or not result.is_file():
        raise _hold(
            "fresh cold subprocess failed: "
            + (
                completed.stderr[-2000:]
                or completed.stdout[-2000:]
                or f"exit {completed.returncode}"
            )
        )
    return _json(result)


def finalize(*, cell_receipts: Sequence[Path], result: Path) -> dict[str, Any]:
    if result.exists() or len(cell_receipts) != 4:
        raise _hold("finalizer requires four receipts and refuses overwrite")
    receipts = [_json(path) for path in cell_receipts]
    names = [str(receipt.get("cell", {}).get("name", "")) for receipt in receipts]
    if set(names) != {"m-qp", "m-ce", "g-qp", "g-ce"}:
        raise _hold("finalizer cell receipt set drift")
    bindings = [receipt.get("bindings") for receipt in receipts]
    if any(binding != bindings[0] for binding in bindings[1:]):
        raise _hold("finalizer immutable binding drift")
    preflights = [receipt.get("preflight") for receipt in receipts]
    if any(item != preflights[0] for item in preflights[1:]) or not isinstance(
        preflights[0], dict
    ):
        raise _hold("finalizer preflight identity drift")
    statuses = []
    for receipt in receipts:
        cell = receipt["cell"]
        outcome = cell["result"][cell["name"].split("-", 1)[1]]
        warm, cold = (
            dict(outcome.get("warm", {})),
            dict(outcome.get("cold", {})).get("gate", {}),
        )
        statuses.append(
            {
                "cell": cell["name"],
                "warm_passed": warm.get("passed"),
                "cold_passed": dict(outcome.get("cold", {})).get("cold_passed"),
                "warm_route_sha256": warm.get("generated_token_ids_sha256"),
                "cold_route_sha256": cold.get("generated_token_ids_sha256"),
                "warm_ledger_sha256": warm.get("ledger_sha256"),
                "cold_ledger_sha256": cold.get("ledger_sha256"),
                "recipe_complete": receipt.get("status") == "frozen_recipe_negative",
                "outcome": receipt.get("status"),
                "classification": receipt.get("status"),
            }
        )
    final = _finalize_status(statuses)
    _atomic_json(
        result,
        {
            "schema_version": SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "cells": names,
            "preflight": preflights[0],
            "results": statuses,
            **final,
        },
    )
    return {**final, "cells": names, "results": statuses}


def _cell(
    model: Any,
    tokenizer: Any,
    native: Mapping[str, Any],
    ordinary_route: Sequence[int],
    ordinary: Mapping[str, torch.Tensor],
    route: Sequence[int],
    label: str,
    optimizer_kind: str,
    binding: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    head = model.get_output_embeddings()
    capture = _capture_route(
        model=model,
        output_head=head,
        native_inputs=native,
        route_tokens=route,
        pad_token_id=int(getattr(tokenizer, "pad_token_id", 0)),
    )
    states = _states(capture, route)
    if not states:
        raise _hold(f"{label}: empty positive set")
    protected_hidden, basis, info = _basis(
        ordinary_route, ordinary, route, capture, states
    )
    selected = list(binding["s_union"])
    norms = _row_norms(head, selected)
    surface = {
        "s_union": selected,
        "s_union_sha256": binding["s_union_sha256"],
        "positive_count": len(states),
        "rank": basis.shape[1],
        "protected_count": len(protected_hidden),
        "protected_sha256": info["protected_hidden_sha256"],
        "basis_sha256": info["basis_sha256"],
        "variable_count": len(selected) * basis.shape[1],
        "parameterization": "D_s=||W_s|| X_s B^T; X_only_trainable_fp64",
        "preflight": binding["preflight"],
    }
    result = {"surface": surface}
    outcome = (
        _qp(states, basis, norms, selected)
        if optimizer_kind == "qp"
        else _ce(
            _all_states(capture, route) if capture else states,
            states,
            basis,
            norms,
            selected,
            warm_evaluate=lambda rows: _warm_gate(
                model,
                tokenizer,
                native,
                head,
                selected,
                rows,
                binding,
                f"matched-{label}-{optimizer_kind}-warm",
            ),
        )
    )
    result[optimizer_kind] = outcome
    candidate = (
        outcome.get("selected", (None, None, None))[2]
        if optimizer_kind == "ce" and outcome.get("selected")
        else (
            outcome
            if outcome.get("feasible")
            and outcome.get("normalized_norm", math.inf) <= NORM_CAP
            else None
        )
    )
    # CE's teacher-forced margin is diagnostic only; ordinary greedy selects it.
    if not candidate or (optimizer_kind == "qp" and not candidate.get("passed", False)):
        return result
    rows = np.asarray(candidate["rows"], dtype=np.float64)
    null = float(np.max(np.abs(protected_hidden @ rows.T), initial=0.0))
    if null > 1e-10:
        raise _hold(f"{label}+{optimizer_kind}: protected-null violation")
    warm = (
        candidate.get("warm")
        if optimizer_kind == "ce"
        else _warm_gate(
            model,
            tokenizer,
            native,
            head,
            selected,
            rows,
            binding,
            f"matched-{label}-{optimizer_kind}-warm",
        )
    )
    if not isinstance(warm, Mapping):
        raise _hold(f"{label}+{optimizer_kind}: warm gate missing")
    result[optimizer_kind]["warm"] = warm
    if not warm["passed"]:
        return result
    # The payload is a warm witness only; promotion requires fresh-process cold parity.
    payload = _save_payload(
        output, f"{label}_{optimizer_kind}", selected, rows, surface
    )
    result[optimizer_kind]["payload"] = payload
    return result


def run(
    *, run_id: str, cell: str, preflight_run_id: str, preflight_sha256: str
) -> Path:
    output = OUTPUT_ROOT / base._safe_run_id(run_id)
    preflight_receipt, actual_preflight_sha256 = _require_preflight(
        preflight_run_id, preflight_sha256
    )
    started = time.monotonic()
    snapshot = _prepare(output)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "run_id": output.name,
        "status": "technical_hold",
        "scientific_status": "not_run",
        "runner_source_snapshot": snapshot,
        "counts": {"model_loads": 0, "cells": 0, "payloads": 0},
        "preflight": {
            "run_id": base._safe_run_id(preflight_run_id),
            "receipt": str(_preflight_path(preflight_run_id)),
            "sha256": actual_preflight_sha256,
            "identity": preflight_receipt["identity"],
        },
    }
    try:
        _require_one_gpu()
        bound = _static_binding()
        bound["runtime_contract"] = parent._binding_contract()["runtime_contract"]
        bound["preflight"] = receipt["preflight"]
        receipt["bindings"] = _binding_receipt()
        receipt["static_g"] = bound["static_g"]
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        setup = bound["runtime_contract"]["setup"]
        with base.open_backend_session(setup["frontend"].launch) as opened:
            if (
                type(opened) is not base.HFBackendSession
                or opened._model is None
                or opened._tokenizer is None
            ):
                raise _hold("requires concrete FP32 HF backend")
            receipt["counts"]["model_loads"] = 1
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            sentinels = full_root._full_root_nontrainable_sentinels(model)
            static_g = _production_static_g_gate(tokenizer, bound)
            receipt["static_g"]["production_parser_global_matcher"] = {
                "passed": static_g["passed"],
                "debt": static_g["debt"],
            }
            if not static_g["passed"]:
                raise _hold("static G failed the production parser/global matcher gate")
            native, prompts, _g, _m = opened._materialize_native_inputs(
                setup["requests"][:1]
            )
            head = model.get_output_embeddings()
            names, parameters, before, frozen_before = _surface_snapshot(model)
            ordinary_route = _greedy_route(model, native, int(tokenizer.pad_token_id))
            ordinary = _capture_route(
                model=model,
                output_head=head,
                native_inputs=native,
                route_tokens=ordinary_route,
                pad_token_id=int(tokenizer.pad_token_id),
            )
            captures = {
                name: _capture_route(
                    model=model,
                    output_head=head,
                    native_inputs=native,
                    route_tokens=route,
                    pad_token_id=int(tokenizer.pad_token_id),
                )
                for name, route in (("m", bound["m_tokens"]), ("g", bound["g_tokens"]))
            }
            positives = {
                name: _states(capture, route)
                for name, (capture, route) in {
                    "m": (captures["m"], bound["m_tokens"]),
                    "g": (captures["g"], bound["g_tokens"]),
                }.items()
            }
            frozen_union = _freeze_s_union(
                m_positive_target_ids=[s["target_token_id"] for s in positives["m"]],
                g_positive_target_ids=[s["target_token_id"] for s in positives["g"]],
            )
            bound.update(
                {
                    "s_union": frozen_union["ids"],
                    "s_union_sha256": frozen_union["sha256"],
                }
            )
            surfaces = {}
            for name in ("m", "g"):
                protected_hidden, basis, info = _basis(
                    ordinary_route,
                    ordinary,
                    bound[f"{name}_tokens"],
                    captures[name],
                    positives[name],
                )
                surfaces[name] = {
                    "positive_count": len(positives[name]),
                    "positive_rank": int(basis.shape[1]),
                    "protected_count": len(protected_hidden),
                    "protected_sha256": info["protected_hidden_sha256"],
                    "basis_sha256": info["basis_sha256"],
                    "s_union_sha256": frozen_union["sha256"],
                    "s_union_count": frozen_union["count"],
                }
            recaptured = _preflight_identity(
                bound=bound,
                ordinary_route=ordinary_route,
                positives=positives,
                sequence_surfaces=surfaces,
                prompts=prompts,
                start_surface=before,
                frozen_surface=frozen_before,
                sentinels=sentinels,
            )
            if recaptured != preflight_receipt["identity"]:
                raise _hold("preflight identity recapture drift")
            sequence, optimizer = cell.split("-", 1)
            result = _cell(
                model,
                tokenizer,
                native,
                ordinary_route,
                ordinary,
                bound[f"{sequence}_tokens"],
                sequence,
                optimizer,
                bound,
                output,
            )
            receipt["cell"] = {
                "name": cell,
                "result": {"surface": result["surface"], optimizer: result[optimizer]},
            }
            receipt["counts"]["cells"] = 1
            receipt["counts"]["payloads"] = int("payload" in result[optimizer])
            receipt["runtime"] = {"warm": opened.receipt.to_artifact_dict()}
            receipt["scientific_status"] = "warm_only_or_frozen_recipe_negative"
            classification = result[optimizer].get("classification")
            receipt["status"] = (
                "certified_qp_infeasible" if classification == "certified_infeasible"
                else "technical_hold" if classification == "HOLD"
                else "frozen_recipe_negative" if "payload" not in result[optimizer]
                else "technical_hold"
            )
            receipt["scientific_status"] = (
                "certified_qp_infeasible" if receipt["status"] == "certified_qp_infeasible"
                else "technical_hold" if receipt["status"] == "technical_hold" and "payload" not in result[optimizer]
                else receipt["scientific_status"]
            )
            names_after, parameters_after, after, frozen_after = _surface_snapshot(model)
            if (
                names != names_after
                or any(a is not b for a, b in zip(parameters, parameters_after, strict=True))
                or before != after
                or frozen_before != frozen_after
            ):
                raise _hold("warm base surface mutation")
            full_root._assert_full_root_sentinels(model, sentinels)
        payload = result[optimizer].get("payload")
        if payload:
            # Drop every warm-model reference before loading the cold subprocess.
            model = tokenizer = native = head = ordinary = captures = positives = None
            bound = sentinels = parameters = parameters_after = names = names_after = None
            before = after = frozen_before = frozen_after = None
            opened = p = static_g = prompts = _g = _m = surfaces = recaptured = None
            gc.collect()
            torch.cuda.empty_cache()
            cold = _run_cold(Path(snapshot["path"]), payload, output, cell, started)
            result[optimizer]["cold"] = cold
            receipt["cell"]["result"][optimizer] = result[optimizer]
            receipt["counts"]["model_loads"] = 2
            receipt["runtime"]["cold"] = cold.get("runtime")
            cold_resources = dict(cold.get("resources", {}))
            if (
                int(cold_resources.get("model_loads", -1)) != 1
                or int(cold_resources.get("peak_cuda_reserved_bytes", 0))
                > RESOURCE_BOUND["max_peak_cuda_reserved_bytes"]
            ):
                raise _hold("cold resource receipt exceeded the frozen bound")
            cold_gate = dict(cold.get("gate", {}))
            warm_gate = dict(result[optimizer].get("warm", {}))
            if (
                cold.get("cold_passed") is True
                and warm_gate.get("generated_token_ids_sha256")
                == cold_gate.get("generated_token_ids_sha256")
                and warm_gate.get("ledger_sha256") == cold_gate.get("ledger_sha256")
            ):
                receipt["status"] = "cold_greedy_41_owner_success"
                receipt["scientific_status"] = "conditional_single_instance_success"
            else:
                receipt["status"] = "technical_hold"
                receipt["scientific_status"] = "warm_only_not_success"
    except BaseException as error:
        receipt["stop"] = {
            "reason": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            _enforce_resources(
                loads=receipt["counts"]["model_loads"], started=started, output=output
            )
        except BaseException as error:
            receipt.update(status="technical_hold", scientific_status="technical_hold",
                           stop={"reason": str(error), "error_type": type(error).__name__})
        receipt["wall_time_seconds"] = time.monotonic() - started
        receipt["resources"] = {
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "artifact_bytes_before_receipt": _artifact_bytes(output),
            "predeclared_bound": RESOURCE_BOUND,
        }
        _atomic_json(output / "receipt.json", receipt)
    return output


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _positive_receipt(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "positive_count": len(states),
        "positive_positions": [int(s["position"]) for s in states],
        "positive_target_ids": [int(s["target_token_id"]) for s in states],
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-bindings", action="store_true")
    p.add_argument("--run-id")
    p.add_argument(
        "--preflight-run-id", help="create the one-GPU preflight receipt only"
    )
    p.add_argument(
        "--preflight-receipt-run-id",
        help="existing preflight receipt run ID required by a cell",
    )
    p.add_argument("--preflight-sha256")
    p.add_argument("--cell", choices=("m-qp", "m-ce", "g-qp", "g-ce"))
    p.add_argument("--cold-verify", action="store_true")
    p.add_argument("--payload", type=Path)
    p.add_argument("--cold-result", type=Path)
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--cell-receipt", action="append", type=Path)
    p.add_argument("--final-result", type=Path)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.check_bindings:
        if any((args.run_id, args.cell, args.cold_verify, args.finalize)):
            raise SystemExit(
                "--check-bindings cannot be combined with execution arguments"
            )
        print(json.dumps(_binding_receipt(), indent=2, sort_keys=True))
        return
    if args.preflight_run_id:
        if any(
            (
                args.run_id,
                args.preflight_receipt_run_id,
                args.preflight_sha256,
                args.cell,
                args.cold_verify,
                args.finalize,
            )
        ):
            raise SystemExit(
                "--preflight-run-id cannot be combined with cell or verifier arguments"
            )
        print(preflight(run_id=args.preflight_run_id))
        return
    if args.cold_verify:
        if not args.payload or not args.cold_result or not args.cell:
            raise SystemExit("--cold-verify requires --payload --cold-result --cell")
        _cold_verify(payload=args.payload, result=args.cold_result, cell=args.cell)
        return
    if args.finalize:
        if not args.cell_receipt or not args.final_result:
            raise SystemExit(
                "--finalize requires four --cell-receipt values and --final-result"
            )
        print(
            json.dumps(
                finalize(cell_receipts=args.cell_receipt, result=args.final_result),
                sort_keys=True,
            )
        )
        return
    if (
        not args.run_id
        or not args.cell
        or not args.preflight_receipt_run_id
        or not args.preflight_sha256
    ):
        raise SystemExit(
            "one-GPU execution requires --run-id --cell --preflight-receipt-run-id --preflight-sha256"
        )
    print(
        run(
            run_id=args.run_id,
            cell=args.cell,
            preflight_run_id=args.preflight_receipt_run_id,
            preflight_sha256=args.preflight_sha256,
        )
    )


if __name__ == "__main__":
    main()
