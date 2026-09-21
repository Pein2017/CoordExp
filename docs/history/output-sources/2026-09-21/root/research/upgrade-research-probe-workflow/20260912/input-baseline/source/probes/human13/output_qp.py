#!/usr/bin/env python3
"""Experiment-local Human13 nested-stage shared output-row QP runner.

This is intentionally a same-panel compiler.  Teacher-forced feasibility is
mechanics evidence; only fresh-process natural greedy verification owns the
stage outcome.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Any

from probes.human13.runtime import prepare_decision_history
from src.qwen.inspection import CaptureHiddenRows



REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/inputs/"
    "human-refined-13.geo_sorted_xy.coord.jsonl"
)
SOURCE_CHECKPOINT = Path(
    "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
    "2026-08-05-closeout/artifacts/training/four-coordinate-xy/"
    "checkpoints/step-2444"
)
BASE_MODEL = Path(
    "/data/Qwen3-VL/model_cache/models/Qwen/"
    "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)
SOURCE_CONFIG = Path(__file__).with_name("configs") / "qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml"
SOURCE_GATE_ROOT = Path("/data/CoordExp/.worktrees/research-probes")
SOURCE_GATE_STUDY = SOURCE_GATE_ROOT / (
    "docs/history/architecture/proposals/2026-06-27-coordexp-infras/"
    "source-studies/special-token-embeddings.md"
)
SOURCE_GATE_RECEIPT = SOURCE_GATE_ROOT / (
    "outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/receipt.json"
)


@dataclass(frozen=True)
class StageSpec:
    image_ids: tuple[int, ...]
    owner_count: int
    decision_state_count: int
    unique_target_token_count: int
    route_token_counts: Mapping[int, int]


STAGES: Mapping[str, StageSpec] = MappingProxyType(
    {
        "N2": StageSpec(
            image_ids=(6040, 16228),
            owner_count=65,
            decision_state_count=592,
            unique_target_token_count=237,
            route_token_counts=MappingProxyType({6040: 136, 16228: 456}),
        ),
        "N4": StageSpec(
            image_ids=(4134, 6040, 13923, 16228),
            owner_count=123,
            decision_state_count=1147,
            unique_target_token_count=401,
            route_token_counts=MappingProxyType(
                {4134: 345, 6040: 136, 13923: 210, 16228: 456}
            ),
        ),
        "N13": StageSpec(
            image_ids=(
                1584,
                2299,
                2685,
                4134,
                5001,
                6040,
                7511,
                10707,
                13348,
                13923,
                14038,
                14439,
                16228,
            ),
            owner_count=392,
            decision_state_count=3637,
            unique_target_token_count=832,
            route_token_counts=MappingProxyType(
                {
                    1584: 172,
                    2299: 415,
                    2685: 282,
                    4134: 345,
                    5001: 212,
                    6040: 136,
                    7511: 400,
                    10707: 184,
                    13348: 138,
                    13923: 210,
                    14038: 438,
                    14439: 249,
                    16228: 456,
                }
            ),
        ),
    }
)
N2_IMAGE_IDS = STAGES["N2"].image_ids
N2_OWNER_COUNT = STAGES["N2"].owner_count
N2_DECISION_STATE_COUNT = STAGES["N2"].decision_state_count
N2_UNIQUE_TARGET_TOKEN_COUNT = STAGES["N2"].unique_target_token_count
N2_ROUTE_TOKEN_COUNTS = STAGES["N2"].route_token_counts
PANEL_IMAGE_COUNT = 13
PANEL_OWNER_COUNT = 392
MARGIN = 0.01
CERTIFICATE_TOLERANCE = 2e-5
MAX_OUTER_SOLVES = 64
MAX_ACTIVE_CONSTRAINTS = 20_000
MAX_ACTIVE_DUAL_BYTES = 512 * 1024 * 1024
MAX_NEW_TOKENS = 3084
IM_END = "<|im_end|>"

EXPECTED_HASHES = {
    PANEL_PATH: "5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23",
    SOURCE_CHECKPOINT / "adapter/adapter_model.safetensors": "49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da",
    SOURCE_CHECKPOINT / "special_token_embeddings/special_token_embeddings.safetensors": "a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2",
    BASE_MODEL / "config.json": "c7d172360d0ff881db59a6f34865c379bbef40d976ad79cfe5fbbf50483655de",
    BASE_MODEL / "tokenizer.json": "ca7e80dee65c629af3b314e76a7587490db3f4e6412df4af9f3b690a9e9916f8",
    SOURCE_CONFIG: "d2217208bc3e419bc9d8c621b4842da262d216780eb1dfc88eade0edf1358f6b",
    SOURCE_GATE_STUDY: "e024f8f9754475cfa6ed81136eae6c72becd2aca53b7b03b9fa3047e8da7d193",
    SOURCE_GATE_RECEIPT: "7da3b22b11ef8a0957bedfe78cac16498313cf724e87cfc921caa0ab7c9f68e4",
}


class HoldError(RuntimeError):
    """A fail-closed mechanics boundary, reported as HOLD by the CLI."""


def _stage_spec(stage: str) -> StageSpec:
    try:
        return STAGES[stage]
    except KeyError as exc:
        raise HoldError(f"unknown stage {stage!r}; expected one of {tuple(STAGES)}") from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _publish_no_clobber(temporary: Path, destination: Path) -> None:
    """Atomically publish one complete same-filesystem file without replacement."""

    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(_canonical_bytes(dict(value)))
        handle.flush()
        os.fsync(handle.fileno())
    _publish_no_clobber(temporary, path)


def load_panel(path: str | Path = PANEL_PATH) -> tuple[dict[str, Any], ...]:
    rows = tuple(json.loads(line) for line in Path(path).read_text().splitlines())
    if any(not isinstance(row, dict) for row in rows):
        raise HoldError("panel rows must be JSON objects")
    return rows


def canonical_route_text(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Serialize the frozen panel order without adding a second ordering rule."""

    objects = row.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("panel row must contain objects")
    serialized: list[str] = []
    for index, obj in enumerate(objects):
        if not isinstance(obj, Mapping):
            raise ValueError(f"object {index} must be a mapping")
        desc = obj.get("desc")
        bbox = obj.get("bbox_2d")
        if not isinstance(desc, str) or not desc:
            raise ValueError(f"object {index} has no description")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(token, str) or not token for token in bbox)
        ):
            raise ValueError(f"object {index} must contain four bbox tokens")
        serialized.append(
            f"<|object_ref_start|>{desc}<|object_ref_end|>"
            f"<|box_start|>{''.join(bbox)}<|box_end|>"
        )
    return "".join(serialized) + IM_END, tuple(serialized)


def tokenize_canonical_route(
    tokenizer: Any, row: Mapping[str, Any]
) -> tuple[tuple[int, ...], dict[str, Any]]:
    text, serialized_rows = canonical_route_text(row)
    ids = tuple(int(value) for value in tokenizer.encode(text, add_special_tokens=False))
    row_ids = tuple(
        tuple(int(value) for value in tokenizer.encode(item, add_special_tokens=False))
        for item in serialized_rows
    )
    terminal_ids = tuple(
        int(value) for value in tokenizer.encode(IM_END, add_special_tokens=False)
    )
    if not ids or any(not item for item in row_ids) or len(terminal_ids) != 1:
        raise HoldError("canonical route tokenization is empty or terminal is not atomic")
    if tuple(value for item in row_ids for value in item) + terminal_ids != ids:
        raise HoldError("canonical row-wise and whole-route tokenization differ")
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    if decoded != text:
        raise HoldError("canonical route tokenizer roundtrip differs")
    return ids, {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_ids_sha256": sha256_json(ids),
        "token_count": len(ids),
        "row_count": len(row_ids),
        "terminal_token_id": terminal_ids[0],
    }


def check_bindings(*, load_tokenizer: bool = True) -> dict[str, Any]:
    from probes.human13.panel import load_frozen_panel
    from src.qwen.special_token_embeddings import (
        load_default_special_token_embedding_source_gate_evidence,
    )

    observed_hashes: dict[str, str] = {}
    for path, expected in EXPECTED_HASHES.items():
        observed = sha256_file(path)
        observed_hashes[str(path)] = observed
        if observed != expected:
            raise HoldError(f"binding hash mismatch: {path}: {observed} != {expected}")
    source_gate = load_default_special_token_embedding_source_gate_evidence(
        SOURCE_GATE_ROOT
    )
    if (
        not source_gate.source_study_passed
        or not source_gate.roundtrip_probe_passed
        or source_gate.probe_receipt is None
    ):
        raise HoldError("special-token embedding source gate is not passed")
    panel = load_panel()
    frozen_panel = load_frozen_panel(PANEL_PATH)
    image_ids = tuple(int(row["image_id"]) for row in panel)
    owner_counts = tuple(len(row["objects"]) for row in panel)
    if len(panel) != PANEL_IMAGE_COUNT or sum(owner_counts) != PANEL_OWNER_COUNT:
        raise HoldError("panel count differs from frozen 13 images / 392 owners")
    if tuple(item.image_id for item in frozen_panel) != image_ids:
        raise HoldError("image-byte-bound panel order differs")
    selected_by_stage: dict[str, tuple[dict[str, Any], ...]] = {}
    for stage, spec in STAGES.items():
        selected = tuple(
            row for row in panel if int(row["image_id"]) in spec.image_ids
        )
        if tuple(int(row["image_id"]) for row in selected) != spec.image_ids:
            raise HoldError(f"{stage} image order differs from frozen registry")
        if sum(len(row["objects"]) for row in selected) != spec.owner_count:
            raise HoldError(f"{stage} owner count differs from {spec.owner_count}")
        selected_by_stage[stage] = selected

    tokenized_by_image: dict[int, tuple[int, ...]] = {}
    route_receipts_by_image: dict[int, dict[str, Any]] = {}
    if load_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, local_files_only=True, trust_remote_code=True
        )
        for row in panel:
            image_id = int(row["image_id"])
            tokenized, receipt = tokenize_canonical_route(tokenizer, row)
            tokenized_by_image[image_id] = tokenized
            route_receipts_by_image[image_id] = {"image_id": image_id, **receipt}
    else:
        for row in panel:
            image_id = int(row["image_id"])
            text, serialized = canonical_route_text(row)
            route_receipts_by_image[image_id] = {
                "image_id": image_id,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "row_count": len(serialized),
            }

    stage_receipts: dict[str, dict[str, Any]] = {}
    for stage, spec in STAGES.items():
        observed_counts = (
            {image_id: len(tokenized_by_image[image_id]) for image_id in spec.image_ids}
            if load_tokenizer
            else None
        )
        if observed_counts != (dict(spec.route_token_counts) if load_tokenizer else None):
            raise HoldError(f"{stage} per-image route token counts differ")
        decision_state_count = (
            sum(observed_counts.values()) if observed_counts is not None else None
        )
        unique_target_count = (
            len(
                set().union(
                    *(set(tokenized_by_image[image_id]) for image_id in spec.image_ids)
                )
            )
            if load_tokenizer
            else None
        )
        if load_tokenizer and decision_state_count != spec.decision_state_count:
            raise HoldError(f"{stage} decision-state count differs")
        if load_tokenizer and unique_target_count != spec.unique_target_token_count:
            raise HoldError(f"{stage} unique target-token count differs")
        stage_receipts[stage] = {
            "image_ids": list(spec.image_ids),
            "owner_count": sum(
                len(row["objects"]) for row in selected_by_stage[stage]
            ),
            "decision_state_count": decision_state_count,
            "unique_target_token_count": unique_target_count,
            "route_token_counts": observed_counts,
            "routes": [route_receipts_by_image[image_id] for image_id in spec.image_ids],
        }

    n2 = stage_receipts["N2"]
    return {
        "schema_version": "human13_output_qp_bindings.v2",
        "producer": {"module": "probes.human13.output_qp",
                     "source_path": str(Path(__file__).resolve()),
                     "source_sha256": sha256_file(__file__)},
        "status": "passed",
        "panel_image_ids": list(image_ids),
        "panel_owner_count": sum(owner_counts),
        "n2_image_ids": list(N2_IMAGE_IDS),
        "n2_owner_count": n2["owner_count"],
        "n2_decision_state_count": n2["decision_state_count"],
        "n2_unique_target_token_count": n2["unique_target_token_count"],
        "stages": stage_receipts,
        "hashes": observed_hashes,
        "embedding_source_gate": {
            "root": str(SOURCE_GATE_ROOT),
            "source_study_passed": source_gate.source_study_passed,
            "roundtrip_probe_passed": source_gate.roundtrip_probe_passed,
            "receipt_ok": source_gate.probe_receipt.get("ok") is True,
            "semantics": source_gate.probe_receipt.get("semantics"),
            "num_selected_tokens": source_gate.probe_receipt.get(
                "num_selected_tokens"
            ),
        },
        "routes": n2["routes"],
    }


class SelectedOutputRowsHook:
    """Add an immutable residual to selected output rows only."""

    def __init__(self, module: Any, selected_token_ids: Any, residual_rows: Any) -> None:
        import torch

        self.module = module
        self.selected_token_ids = torch.as_tensor(selected_token_ids, dtype=torch.long)
        self.residual_rows = torch.as_tensor(residual_rows, dtype=torch.float64)
        if self.residual_rows.ndim != 2:
            raise ValueError("residual_rows must be a matrix")
        if self.residual_rows.shape[0] != self.selected_token_ids.numel():
            raise ValueError("selected ids and residual rows differ")
        if len(set(int(x) for x in self.selected_token_ids.tolist())) != int(
            self.selected_token_ids.numel()
        ):
            raise ValueError("selected token ids must be unique")
        self.handle: Any | None = None
        self.call_count = 0

    def _hook(self, _module: Any, args: tuple[Any, ...], output: Any) -> Any:
        import torch

        if not isinstance(output, torch.Tensor) or not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("output-head hook requires tensor input/output")
        self.call_count += 1
        if self.selected_token_ids.numel() == 0:
            return output
        hidden = args[0]
        if hidden.shape[-1] != self.residual_rows.shape[1]:
            raise RuntimeError("output-head hidden width differs from payload")
        ids = self.selected_token_ids.to(output.device)
        rows = self.residual_rows.to(device=output.device, dtype=hidden.dtype)
        delta = torch.matmul(hidden, rows.transpose(0, 1)).to(output.dtype)
        changed = output.clone()
        changed[..., ids] = changed[..., ids] + delta
        return changed

    def __enter__(self) -> "SelectedOutputRowsHook":
        if self.handle is not None:
            raise RuntimeError("output-head hook already installed")
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc: Any) -> None:
        assert self.handle is not None
        self.handle.remove()
        self.handle = None


def _output_head(model: Any) -> Any:
    head = model.get_output_embeddings()
    if head is None or not callable(getattr(head, "register_forward_hook", None)):
        raise HoldError("model has no hookable output head")
    if head is model.get_input_embeddings():
        raise HoldError("refusing to hook a module shared with input embeddings")
    return head


def recover_fixed_max(
    top_ids: Sequence[int],
    top_logits: Sequence[float],
    selected_ids: set[int],
    *,
    target_id: int,
) -> tuple[int, float]:
    """Return the exact best unchanged competitor from a sufficient top-K."""

    if len(top_ids) != len(top_logits):
        raise ValueError("top ids/logits differ")
    if any(float(left) < float(right) for left, right in zip(top_logits, top_logits[1:])):
        raise ValueError("top-K logits must be sorted descending")
    for token_id, logit in zip(top_ids, top_logits, strict=True):
        token_id = int(token_id)
        if token_id != int(target_id) and token_id not in selected_ids:
            return token_id, float(logit)
    raise HoldError("top-K does not contain an unchanged non-target competitor")


def select_trainable_rows(
    target_ids: Sequence[int],
    top_ids: Any,
    top_logits: Any,
    *,
    target_logits: Sequence[float] | None = None,
    margin: float = MARGIN,
) -> tuple[int, ...]:
    """Select route targets whose exact base full-vocabulary margin is deficient."""

    import numpy as np

    targets = np.asarray(target_ids, dtype=np.int64)
    ids = np.asarray(top_ids, dtype=np.int64)
    logits = np.asarray(top_logits, dtype=np.float64)
    if ids.shape != logits.shape or ids.ndim != 2 or ids.shape[0] != targets.size:
        raise ValueError("top-K evidence shape differs from target positions")
    explicit_targets = (
        None if target_logits is None else np.asarray(target_logits, dtype=np.float64)
    )
    if explicit_targets is not None and explicit_targets.shape != targets.shape:
        raise ValueError("target logits shape differs from target positions")
    selected: set[int] = set()
    for pos, target in enumerate(targets.tolist()):
        competitor = max(
            float(value)
            for token, value in zip(ids[pos], logits[pos], strict=True)
            if int(token) != target
        )
        if explicit_targets is None:
            target_matches = [
                float(value)
                for token, value in zip(ids[pos], logits[pos], strict=True)
                if int(token) == target
            ]
            if len(target_matches) != 1:
                raise HoldError("target logit must be explicit when target is outside top-K")
            target_logit = target_matches[0]
        else:
            target_logit = float(explicit_targets[pos])
        if target_logit - competitor < margin:
            selected.add(target)
    return tuple(sorted(selected))


@dataclass(frozen=True)
class _Constraint:
    position: int
    target_row: int | None
    competitor_row: int | None
    rhs: float
    kind: str
    competitor_token_id: int

    @property
    def key(self) -> tuple[int, int | None, int | None, str]:
        return (self.position, self.target_row, self.competitor_row, self.kind)


def _lhs(constraint: _Constraint, coefficients: Any, state: Any) -> float:
    value = 0.0
    if constraint.target_row is not None:
        value += float(coefficients[constraint.target_row] @ state)
    if constraint.competitor_row is not None:
        value -= float(coefficients[constraint.competitor_row] @ state)
    return value


def solve_minimum_frobenius(
    *,
    hidden_states: Any,
    target_ids: Sequence[int],
    route_token_ids: Sequence[int],
    base_route_logits: Any,
    top_ids: Any,
    top_logits: Any,
    margin: float = MARGIN,
    certificate_tolerance: float = CERTIFICATE_TOLERANCE,
    max_outer_solves: int = MAX_OUTER_SOLVES,
    max_active_constraints: int = MAX_ACTIVE_CONSTRAINTS,
) -> dict[str, Any]:
    """Solve the selected-row QP with an exhaustive cutting-plane separator."""

    import numpy as np
    from scipy.optimize import minimize

    hidden = np.asarray(hidden_states, dtype=np.float64)
    targets = np.asarray(target_ids, dtype=np.int64)
    route_ids = np.asarray(route_token_ids, dtype=np.int64)
    route_logits = np.asarray(base_route_logits, dtype=np.float64)
    top_token_ids = np.asarray(top_ids, dtype=np.int64)
    top_values = np.asarray(top_logits, dtype=np.float64)
    if hidden.ndim != 2 or hidden.shape[0] != targets.size:
        raise ValueError("hidden states differ from target positions")
    if route_logits.shape != (targets.size, route_ids.size):
        raise ValueError("route logits shape differs")
    if top_token_ids.shape != top_values.shape or top_values.shape[0] != targets.size:
        raise ValueError("top-K shape differs")
    if len(set(route_ids.tolist())) != route_ids.size:
        raise ValueError("route token ids must be unique")
    if top_values.shape[1] <= route_ids.size:
        raise HoldError("K must exceed the number of unique route targets")
    route_index = {int(token): index for index, token in enumerate(route_ids)}
    if any(int(token) not in route_index for token in targets):
        raise HoldError("target token is absent from route-logit evidence")
    target_base_logits = np.asarray(
        [route_logits[pos, route_index[int(target)]] for pos, target in enumerate(targets)],
        dtype=np.float64,
    )
    selected_ids = select_trainable_rows(
        targets,
        top_token_ids,
        top_values,
        target_logits=target_base_logits,
        margin=margin,
    )
    selected_set = set(selected_ids)
    selected_index = {token: index for index, token in enumerate(selected_ids)}
    if not selected_ids:
        raise HoldError("no selected rows: zero residual already satisfies teacher-forced margins")

    _, singular, vh = np.linalg.svd(hidden, full_matrices=False)
    if singular.size == 0 or singular[0] == 0.0:
        raise HoldError("captured hidden states have zero span")
    rank_tolerance = max(hidden.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.sum(singular > rank_tolerance))
    basis = vh[:rank]
    projected = hidden @ basis.T

    fixed: list[tuple[int, float]] = []
    for pos, target in enumerate(targets.tolist()):
        fixed.append(
            recover_fixed_max(
                top_token_ids[pos],
                top_values[pos],
                selected_set,
                target_id=target,
            )
        )

    deficient_position_count = sum(
        1
        for pos, target in enumerate(targets.tolist())
        if target_base_logits[pos]
        - max(
            float(value)
            for token, value in zip(top_token_ids[pos], top_values[pos], strict=True)
            if int(token) != target
        )
        < margin
    )

    def base(token: int, pos: int) -> float:
        return float(route_logits[pos, route_index[token]])

    def candidate_constraints(pos: int) -> list[_Constraint]:
        target = int(targets[pos])
        target_row = selected_index.get(target)
        result: list[_Constraint] = []
        for competitor in selected_ids:
            if competitor == target:
                continue
            result.append(
                _Constraint(
                    pos,
                    target_row,
                    selected_index[competitor],
                    margin - base(target, pos) + base(competitor, pos),
                    "selected",
                    competitor,
                )
            )
        # A fixed target was not deficient at base and all fixed rows remain
        # unchanged.  Only selected competitors can threaten it after fitting.
        if target_row is not None:
            fixed_id, fixed_logit = fixed[pos]
            result.append(
                _Constraint(
                    pos,
                    target_row,
                    None,
                    margin - base(target, pos) + fixed_logit,
                    "fixed_max",
                    fixed_id,
                )
            )
        return result

    active: list[_Constraint] = []
    active_keys: set[tuple[int, int | None, int | None, str]] = set()
    coefficients = np.zeros((len(selected_ids), rank), dtype=np.float64)
    dual = np.zeros(0, dtype=np.float64)
    outer_solve_count = 0
    optimizer_iterations = 0

    def separate(current: Any) -> tuple[float, list[_Constraint], _Constraint | None]:
        worst = -math.inf
        worst_constraint: _Constraint | None = None
        additions: list[_Constraint] = []
        for pos in range(targets.size):
            state_worst = -math.inf
            state_constraint: _Constraint | None = None
            for constraint in candidate_constraints(pos):
                violation = constraint.rhs - _lhs(constraint, current, projected[pos])
                if violation > worst:
                    worst, worst_constraint = violation, constraint
                if violation > state_worst:
                    state_worst, state_constraint = violation, constraint
            if (
                state_constraint is not None
                and state_worst > certificate_tolerance
                and state_constraint.key not in active_keys
            ):
                additions.append(state_constraint)
        return worst, additions, worst_constraint

    while True:
        worst, additions, largest = separate(coefficients)
        if worst <= certificate_tolerance:
            break
        if outer_solve_count >= max_outer_solves:
            raise HoldError("cutting-plane outer solve limit reached")
        if not additions:
            raise HoldError("violated constraints remain but no cutting plane was added")
        if len(active) + len(additions) > max_active_constraints:
            raise HoldError("active constraint ceiling reached before certificate")
        active.extend(additions)
        active_keys.update(item.key for item in additions)
        if len(active) * 8 > MAX_ACTIVE_DUAL_BYTES:
            raise HoldError("active dual allocation ceiling reached")
        start = np.pad(dual, (0, len(active) - dual.size))

        def reconstruct(lambdas: Any) -> Any:
            current = np.zeros_like(coefficients)
            for weight, constraint in zip(lambdas, active, strict=True):
                if weight == 0.0:
                    continue
                z = projected[constraint.position]
                if constraint.target_row is not None:
                    current[constraint.target_row] += weight * z
                if constraint.competitor_row is not None:
                    current[constraint.competitor_row] -= weight * z
            return current

        def objective(lambdas: Any) -> tuple[float, Any]:
            current = reconstruct(lambdas)
            value = 0.5 * float(np.sum(current * current)) - float(
                np.dot(lambdas, [item.rhs for item in active])
            )
            gradient = np.asarray(
                [
                    _lhs(item, current, projected[item.position]) - item.rhs
                    for item in active
                ],
                dtype=np.float64,
            )
            return value, gradient

        outer_solve_count += 1
        optimizer_ftol = 1e-14
        for segment_index in range(1, 4):
            polish = optimizer_ftol == 0.0
            result = minimize(
                objective,
                start,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0.0, None)] * len(active),
                options={
                    "ftol": optimizer_ftol,
                    "gtol": 1e-10,
                    "maxiter": 4000,
                    "maxls": 50,
                },
            )
            optimizer_iterations += int(result.nit)
            candidate_dual = np.asarray(result.x, dtype=np.float64)
            candidate_coefficients = reconstruct(candidate_dual)
            _, candidate_gradient = objective(candidate_dual)
            candidate_primal = 0.5 * float(
                np.sum(candidate_coefficients * candidate_coefficients)
            )
            candidate_dual_objective = float(
                np.dot(candidate_dual, [item.rhs for item in active])
            ) - candidate_primal
            candidate_worst_violation, candidate_additions, _ = separate(
                candidate_coefficients
            )
            progress = {
                "schema": "human13_output_qp_solver_progress.v1",
                "diagnostic_outer_solve_index": outer_solve_count,
                "diagnostic_segment_index": segment_index,
                "diagnostic_segment_limit": 3,
                "diagnostic_polish": polish,
                "diagnostic_optimizer_ftol": optimizer_ftol,
                "diagnostic_active_constraint_count": len(active),
                "diagnostic_active_set_sha256": sha256_json([item.key for item in active]),
                "diagnostic_result_success": bool(result.success),
                "diagnostic_result_status": int(result.status),
                "diagnostic_result_message": str(result.message),
                "diagnostic_result_nit": int(result.nit),
                "diagnostic_result_nfev": int(result.nfev),
                "diagnostic_result_fun": float(result.fun),
                "diagnostic_lambda_l2_norm": float(np.linalg.norm(candidate_dual)),
                "diagnostic_lambda_max": float(np.max(candidate_dual)),
                "diagnostic_projected_gradient_kkt_inf_norm": float(
                    np.linalg.norm(
                        candidate_dual
                        - np.maximum(0.0, candidate_dual - candidate_gradient),
                        ord=np.inf,
                    )
                ),
                "diagnostic_full_registered_worst_primal_violation": float(
                    candidate_worst_violation
                ),
                "diagnostic_primal_half_frobenius_squared": candidate_primal,
                "diagnostic_dual_objective": candidate_dual_objective,
                "diagnostic_candidate_gap": candidate_primal - candidate_dual_objective,
                "diagnostic_active_complementarity_max": float(
                    np.max(np.abs(candidate_dual * candidate_gradient))
                ),
            }
            if hasattr(result, "njev"):
                progress["diagnostic_result_njev"] = int(result.njev)
            print(
                json.dumps(progress, sort_keys=True, separators=(",", ":")),
                flush=True,
            )
            if result.success:
                if (
                    candidate_worst_violation <= certificate_tolerance
                    or candidate_additions
                    or segment_index == 3
                ):
                    break
                start = candidate_dual
                optimizer_ftol = 0.0
                continue
            if (
                int(result.status) == 1
                and "TOTAL NO. OF ITERATIONS REACHED LIMIT" in str(result.message)
                and segment_index < 3
            ):
                start = candidate_dual
                continue
            raise HoldError(f"QP dual optimizer failed: {result.message}")
        dual = candidate_dual
        coefficients = candidate_coefficients

    residual_rows = coefficients @ basis
    # Replay the actual hook arithmetic boundary: FP64 payload rows are cast to
    # FP32 and multiplied by FP32 hidden states before addition to FP32 logits.
    fp32_delta = (
        hidden.astype(np.float32) @ residual_rows.astype(np.float32).T
    ).astype(np.float32)
    minimum_fp32_margin = math.inf
    worst_fp32: dict[str, Any] | None = None
    for pos, target in enumerate(targets.tolist()):
        target_delta = (
            float(fp32_delta[pos, selected_index[target]]) if target in selected_set else 0.0
        )
        target_logit = np.float32(base(target, pos)) + np.float32(target_delta)
        for competitor in selected_ids:
            if competitor == target:
                continue
            margin_value = float(
                target_logit
                - (
                    np.float32(base(competitor, pos))
                    + np.float32(fp32_delta[pos, selected_index[competitor]])
                )
            )
            if margin_value < minimum_fp32_margin:
                minimum_fp32_margin = margin_value
                worst_fp32 = {"position": pos, "competitor_token_id": competitor, "kind": "selected"}
        if target in selected_set:
            fixed_id, fixed_logit = fixed[pos]
            margin_value = float(target_logit - np.float32(fixed_logit))
            if margin_value < minimum_fp32_margin:
                minimum_fp32_margin = margin_value
                worst_fp32 = {"position": pos, "competitor_token_id": fixed_id, "kind": "fixed_max"}
    if minimum_fp32_margin < margin - certificate_tolerance:
        raise HoldError(
            f"FP32 hook replay misses registered margin: {minimum_fp32_margin}"
        )

    primal = 0.5 * float(np.sum(coefficients * coefficients))
    dual_objective = float(np.dot(dual, [item.rhs for item in active])) - primal
    row_energy = np.sum(residual_rows * residual_rows, axis=1)
    sv_residual = np.linalg.svd(residual_rows, compute_uv=False)
    energy = sv_residual * sv_residual
    effective_rank = (
        float(energy.sum() ** 2 / np.sum(energy * energy)) if np.any(energy) else 0.0
    )
    rank_95 = (
        int(np.searchsorted(np.cumsum(energy) / energy.sum(), 0.95) + 1)
        if np.any(energy)
        else 0
    )
    worst, _, _ = separate(coefficients)
    largest_required = max(
        (constraint for pos in range(targets.size) for constraint in candidate_constraints(pos)),
        key=lambda item: item.rhs,
    )
    return {
        "selected_token_ids": np.asarray(selected_ids, dtype=np.int64),
        "residual_rows": np.asarray(residual_rows, dtype=np.float64),
        "receipt": {
            "position_count": int(targets.size),
            "deficient_position_count": deficient_position_count,
            "selected_row_count": len(selected_ids),
            "hidden_span_rank": rank,
            "free_variable_count": len(selected_ids) * rank,
            "hidden_span_singular_values": singular.tolist(),
            "registered_constraint_count": sum(
                len(candidate_constraints(pos)) for pos in range(targets.size)
            ),
            "active_constraint_count": len(active),
            "outer_solve_count": outer_solve_count,
            "optimizer_iteration_count": optimizer_iterations,
            "max_fp64_violation": float(worst),
            "minimum_primal_slack": float(-worst),
            "minimum_fp32_hook_margin": float(minimum_fp32_margin),
            "certificate_tolerance": certificate_tolerance,
            "full_vocab_partition_certificate": {
                "top_k": int(top_values.shape[1]),
                "unique_route_target_count": int(route_ids.size),
                "k_exceeds_unique_targets": bool(top_values.shape[1] > route_ids.size),
                "selected_rows_are_route_targets": selected_set.issubset(set(route_ids.tolist())),
                "target_excluded_from_competitors": True,
                "worst_fp32_constraint": worst_fp32,
            },
            "objective_half_frobenius_squared": primal,
            "dual_objective": dual_objective,
            "duality_gap": primal - dual_objective,
            "residual_rank": int(np.sum(sv_residual > (sv_residual[0] * 1e-12))) if sv_residual.size else 0,
            "residual_effective_rank": effective_rank,
            "residual_rank_95_percent_energy": rank_95,
            "residual_first_direction_energy_share": (
                float(energy[0] / energy.sum()) if np.any(energy) else 0.0
            ),
            "largest_row_energy_share": float(row_energy.max() / row_energy.sum()) if row_energy.sum() else 0.0,
            "largest_required_margin_constraint": {
                "position": largest_required.position,
                "kind": largest_required.kind,
                "competitor_token_id": largest_required.competitor_token_id,
                "rhs": largest_required.rhs,
            },
        },
    }


def _runtime_setup_images(
    image_ids: Sequence[int],
    adapter_path: Path | None = None,
) -> tuple[Any, Any, tuple[Any, ...], Any, Any]:
    """Load one exact Source surface and the ordered selected requests."""

    from probes.human13.runtime import (
        derive_hf_fp32_sdpa_batch_one_launch,
        validate_hf_fp32_sdpa_batch_one,
    )
    from probes.human13.runtime import (
        _build_requests,
        physical_image_id,
    )
    from src.config.fingerprint import sha256_json as config_sha256_json
    from src.config.inference import InferConfig, load_research_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import open_backend_session
    from src.inference.hf_backend import open_hf_backend_session
    from probes.human13.runtime import load_source_components
    from src.inference.runtime import assemble_frontend

    payload = load_research_infer_config(SOURCE_CONFIG).config.model_dump(mode="json")
    payload["adapter"]["path"] = str(
        SOURCE_CHECKPOINT / "adapter" if adapter_path is None else adapter_path
    )
    payload["embedding_delta"]["path"] = str(
        SOURCE_CHECKPOINT / "special_token_embeddings"
    )
    payload["embedding_delta"]["source_gate_root"] = str(SOURCE_GATE_ROOT)
    payload["run"]["artifact_root"] = f"/tmp/human13-output-qp-{os.getpid()}"
    payload["generation"].update(
        {
            "batch_size": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "repetition_penalty": 1.0,
        }
    )
    payload["debug"].update({"smoke": True, "dry_run": False})
    config = InferConfig.model_validate(payload)
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=config_sha256_json(
            config.generation.model_dump(mode="json")
        ),
    )
    examples = tuple(load_raw_examples(config.data.input_jsonl))
    requests = _build_requests(config, frontend, examples)
    selected_by_image: dict[int, list[tuple[Any, Any]]] = {
        image_id: [] for image_id in image_ids
    }
    for example, request in zip(examples, requests, strict=True):
        image_id = int(physical_image_id(example))
        if image_id in selected_by_image:
            selected_by_image[image_id].append((example, request))
    for image_id, selected in selected_by_image.items():
        if len(selected) != 1:
            raise HoldError(f"image {image_id} resolved to {len(selected)} requests")
    launch = derive_hf_fp32_sdpa_batch_one_launch(frontend.launch)
    loaded = load_source_components(launch)
    session_context = open_backend_session(
        launch,
        opener=lambda requested: open_hf_backend_session(
            requested, components_loader=lambda _: loaded
        ),
    )
    session = session_context.__enter__()
    try:
        runtime_identity = validate_hf_fp32_sdpa_batch_one(
            derive_hf_fp32_sdpa_batch_one_launch(frontend.launch), session.receipt
        )
    except BaseException:
        session_context.__exit__(*sys.exc_info())
        raise
    return (
        session_context,
        session,
        tuple(selected_by_image[image_id][0] for image_id in image_ids),
        runtime_identity,
        loaded.qwen,
    )


def _runtime_setup(image_id: int) -> tuple[Any, Any, Any, Any, Any]:
    """Load the exact Source surface and one selected request."""

    context, session, selected, runtime_identity, components = _runtime_setup_images((image_id,))
    return context, session, selected[0], runtime_identity, components


def _decode(session: Any, request: Any, *, repetition_penalty: float) -> Any:
    from src.inference.backend import GenerationPolicy

    return tuple(
        session.decode(
            (
                replace(
                    request,
                    generation_policy=GenerationPolicy(
                        max_new_tokens=MAX_NEW_TOKENS,
                        repetition_penalty=float(repetition_penalty),
                        temperature=0.0,
                        top_p=1.0,
                    ),
                ),
            )
        )
    )[0]


def _result_identity(result: Any) -> dict[str, Any]:
    token_ids = tuple(int(value) for value in result.generated_token_ids)
    parser_text = str(result.parser_text)
    return {
        "generated_token_ids_sha256": sha256_json(token_ids),
        "parser_text_sha256": hashlib.sha256(parser_text.encode()).hexdigest(),
        "stop_reason": str(result.stop_reason),
        "generated_token_count": len(token_ids),
    }


def _natural_order_violation_row_count(
    boxes: Sequence[tuple[float, float, float, float]],
) -> int:
    """Count emitted (x1, y1) anchors below the running prior maximum."""

    prior: tuple[float, float] | None = None
    violations = 0
    for box in boxes:
        anchor = box[0], box[1]
        if prior is not None and anchor < prior:
            violations += 1
        prior = anchor if prior is None else max(prior, anchor)
    return violations


def _save_tensors_immutable(path: Path, tensors: Mapping[str, Any]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(dict(tensors), str(temporary))
        _publish_no_clobber(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_image(
    *, image_id: int, output_dir: Path, stage: str = "N2"
) -> dict[str, Any]:
    import torch

    spec = _stage_spec(stage)
    if image_id not in spec.image_ids:
        raise HoldError(f"capture image must be one of {spec.image_ids} for {stage}")
    bindings = check_bindings(load_tokenizer=False)
    panel = {int(row["image_id"]): row for row in load_panel()}
    started = time.perf_counter()
    context, session, (example, request), runtime_identity, components = _runtime_setup(image_id)
    try:
        tokenizer = components.tokenizer
        route_ids, route_receipt = tokenize_canonical_route(tokenizer, panel[image_id])
        if len(route_ids) != spec.route_token_counts[image_id]:
            raise HoldError("per-image canonical decision count differs")
        all_route_ids: set[int] = set()
        for registered_id in spec.image_ids:
            registered, _ = tokenize_canonical_route(tokenizer, panel[registered_id])
            all_route_ids.update(registered)
        route_vocab_ids = tuple(sorted(all_route_ids))
        if len(route_vocab_ids) != spec.unique_target_token_count:
            raise HoldError(f"{stage} unique canonical target-token count differs")
        top_k = len(route_vocab_ids) + 1
        source_rp10 = _decode(session, request, repetition_penalty=1.0)
        source_rp110 = _decode(session, request, repetition_penalty=1.10)
        head = _output_head(components.model)
        empty_rows = torch.empty((0, int(head.weight.shape[1])), dtype=torch.float64)
        with SelectedOutputRowsHook(head, [], empty_rows) as zero_hook:
            zero_result = _decode(session, request, repetition_penalty=1.0)
        if _result_identity(zero_result) != _result_identity(source_rp10):
            raise HoldError("zero-residual token/parser parity differs from Source")
        model_inputs, executed_ids, decision_positions = prepare_decision_history(
            components, request, route_ids
        )
        with CaptureHiddenRows(
            head, range(len(route_ids)), boundary="input", dtype=torch.float64
        ) as state_capture:
            with torch.inference_mode():
                output = components.model(**model_inputs)
        logits = output.logits[0]
        if tuple(logits.shape[:1]) != (len(route_ids),):
            raise HoldError("position-selective logits do not cover the canonical route")
        route_tensor = torch.tensor(route_vocab_ids, device=logits.device, dtype=torch.long)
        top_values, top_ids = torch.topk(logits.float(), k=top_k, dim=-1)
        base_route_logits = logits.float().index_select(-1, route_tensor)
        target_tensor = torch.tensor(route_ids, dtype=torch.long)
        tensor_path = output_dir / f"capture-{image_id}.safetensors"
        receipt_path = output_dir / f"capture-{image_id}.json"
        tensors = {
            "hidden_states": state_capture.hidden.contiguous(),
            "target_token_ids": target_tensor,
            "route_token_ids": torch.tensor(route_vocab_ids, dtype=torch.long),
            "base_route_logits": base_route_logits.detach().cpu().contiguous(),
            "top_token_ids": top_ids.detach().cpu().to(torch.long).contiguous(),
            "top_logits": top_values.detach().cpu().contiguous(),
        }
        _save_tensors_immutable(tensor_path, tensors)
        receipt = {
            "schema_version": "human13_output_qp_capture.v1",
            "status": "captured",
            "stage": stage,
            "image_ids": list(spec.image_ids),
            "image_id": image_id,
            "owner_count": len(panel[image_id]["objects"]),
            "route": route_receipt,
            "top_k": top_k,
            "unique_route_target_count": len(route_vocab_ids),
            **(
                {"unique_n2_route_token_count": len(route_vocab_ids)}
                if stage == "N2"
                else {}
            ),
            "route_vocab_token_ids_sha256": sha256_json(route_vocab_ids),
            "source_rp1_0": _result_identity(source_rp10),
            "source_rp1_10_monitor": _result_identity(source_rp110),
            "zero_residual_rp1_0": _result_identity(zero_result),
            "zero_hook_call_count": zero_hook.call_count,
            "runtime_identity": runtime_identity,
            "binding_receipt_sha256": sha256_json(bindings),
            "capture_tensor": str(tensor_path.resolve()),
            "capture_tensor_sha256": sha256_file(tensor_path),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
            "process_id": os.getpid(),
            "model_load_count": 1,
            "natural_generation_count": 3,
            "teacher_forced_capture_count": 1,
        }
        immutable_json(receipt_path, receipt)
        return {**receipt, "receipt_path": str(receipt_path.resolve())}
    finally:
        context.__exit__(None, None, None)


def solve_captures(
    *, captures: Sequence[Path], output_dir: Path, stage: str = "N2"
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    spec = _stage_spec(stage)
    if len(captures) != len(spec.image_ids):
        raise HoldError(
            f"solve {stage} requires exactly {len(spec.image_ids)} capture tensors"
        )
    receipts: list[dict[str, Any]] = []
    tensors: list[dict[str, Any]] = []
    owner_counts = {
        int(row["image_id"]): len(row["objects"])
        for row in load_panel()
        if int(row["image_id"]) in spec.image_ids
    }
    for expected_image, path in zip(spec.image_ids, captures, strict=True):
        receipt_path = path.with_suffix(".json")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("stage") != stage or tuple(receipt.get("image_ids", ())) != spec.image_ids:
            raise HoldError("capture stage receipt differs from requested stage")
        if int(receipt.get("image_id", -1)) != expected_image:
            raise HoldError(f"capture order differs from frozen {stage} order")
        if receipt.get("capture_tensor_sha256") != sha256_file(path):
            raise HoldError("capture tensor hash differs from receipt")
        tensor = load_file(str(path))
        expected_states = spec.route_token_counts[expected_image]
        if (
            int(receipt.get("owner_count", -1)) != owner_counts[expected_image]
            or int(receipt.get("route", {}).get("token_count", -1)) != expected_states
            or int(receipt.get("top_k", -1)) != spec.unique_target_token_count + 1
            or int(receipt.get("unique_route_target_count", -1))
            != spec.unique_target_token_count
            or int(tensor["target_token_ids"].shape[0]) != expected_states
        ):
            raise HoldError("capture receipt counts differ from stage registry")
        receipts.append(receipt)
        tensors.append(tensor)
    route_ids = tensors[0]["route_token_ids"]
    if any(not torch.equal(item["route_token_ids"], route_ids) for item in tensors[1:]):
        raise HoldError("capture route-vocabulary identities differ")
    route_vocab_sha = sha256_json(tuple(int(value) for value in route_ids.tolist()))
    if (
        int(route_ids.numel()) != spec.unique_target_token_count
        or any(
            receipt.get("route_vocab_token_ids_sha256") != route_vocab_sha
            for receipt in receipts
        )
    ):
        raise HoldError("capture route-vocabulary receipt binding differs")
    combined = {
        name: torch.cat([item[name] for item in tensors], dim=0).numpy()
        for name in (
            "hidden_states",
            "target_token_ids",
            "base_route_logits",
            "top_token_ids",
            "top_logits",
        )
    }
    if int(combined["target_token_ids"].shape[0]) != spec.decision_state_count:
        raise HoldError(
            f"combined capture does not contain {spec.decision_state_count} decision states"
        )
    solved = solve_minimum_frobenius(
        hidden_states=combined["hidden_states"],
        target_ids=combined["target_token_ids"],
        route_token_ids=route_ids.numpy(),
        base_route_logits=combined["base_route_logits"],
        top_ids=combined["top_token_ids"],
        top_logits=combined["top_logits"],
    )
    payload_path = output_dir / f"human13-{stage.lower()}-output-residual.safetensors"
    receipt_path = payload_path.with_suffix(".json")
    _save_tensors_immutable(
        payload_path,
        {
            "selected_token_ids": torch.from_numpy(solved["selected_token_ids"]),
            "residual_rows": torch.from_numpy(solved["residual_rows"]),
        },
    )
    receipt = {
        "schema_version": "human13_output_qp_payload.v1",
        "status": "solved_teacher_forced_mechanics_only",
        "stage": stage,
        "image_ids": list(spec.image_ids),
        "owner_count": spec.owner_count,
        "margin": MARGIN,
        "payload_path": str(payload_path.resolve()),
        "payload_sha256": sha256_file(payload_path),
        "selected_token_ids": solved["selected_token_ids"].tolist(),
        "solver": solved["receipt"],
        "captures": [
            {
                "image_id": receipt["image_id"],
                "tensor_sha256": receipt["capture_tensor_sha256"],
                "pre_source_rp1_0": receipt["source_rp1_0"],
            }
            for receipt in receipts
        ],
        "artifact_bytes": payload_path.stat().st_size,
    }
    immutable_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path.resolve())}


def _evaluate_result(
    *,
    result: Any,
    image_id: int,
    canonical_ids: Sequence[int],
    backend_version: str,
    repetition_penalty: float,
) -> dict[str, Any]:
    from probes.human13.evidence import (
        source_request,
        trajectory_input_from_decode_result,
    )
    from probes.human13.panel import load_frozen_panel
    from src.eval.assignment import global_matches as _global_matches
    from src.data.geometry import iou_xyxy
    from src.eval.detection_categories import normalize_coco_category_name

    panel = {row.image_id: row for row in load_frozen_panel()}
    frozen = panel[image_id]
    raw = next(row for row in load_panel() if int(row["image_id"]) == image_id)
    scientific = source_request(
        backend_version=backend_version, repetition_penalty=float(repetition_penalty)
    )
    trajectory, parse = trajectory_input_from_decode_result(
        image_id=image_id,
        trajectory_id=f"output-qp-verify:{image_id}",
        request=scientific,
        result=result,
        image_width=int(raw["width"]),
        image_height=int(raw["height"]),
    )
    retained: list[tuple[str, tuple[float, float, float, float]]] = []
    emitted_boxes: list[tuple[float, float, float, float]] = []
    duplicate_count = 0
    for prediction in trajectory.rows:
        category = normalize_coco_category_name(prediction.category)
        box_values = [float(value) for value in prediction.bbox]
        if len(box_values) != 4:
            raise HoldError("parsed prediction bbox layout differs")
        box = box_values[0], box_values[1], box_values[2], box_values[3]
        emitted_boxes.append(box)
        if any(iou_xyxy(previous[1], box) > 0.95 for previous in retained):
            duplicate_count += 1
        else:
            retained.append((category, box))
    gt = [
        (normalize_coco_category_name(owner.category), owner.bbox)
        for owner in sorted(
            frozen.owners, key=lambda item: (item.source_object_index, item.owner_id)
        )
    ]
    matches = {
        str(int(threshold * 100)): len(_global_matches(gt, retained, threshold))
        for threshold in (0.50, 0.60, 0.80)
    }
    matched_prediction_count = len(_global_matches(gt, retained, 0.50))
    natural_order_violation_row_count = _natural_order_violation_row_count(
        emitted_boxes
    )
    identity = _result_identity(result)
    generated = tuple(int(value) for value in result.generated_token_ids)
    return {
        **identity,
        "owner_count": len(gt),
        "matched_owner_count": matches,
        "duplicate_count": duplicate_count,
        "unmatched_prediction_count": len(retained) - matched_prediction_count,
        "malformed_count": len(parse["dropped_predictions"]),
        "cap_debt": str(result.stop_reason) not in {"im_end", "eos", "natural_stop"},
        "natural_eos": str(result.stop_reason) in {"im_end", "eos", "natural_stop"},
        "exact_route": generated == tuple(canonical_ids),
        "natural_order_violation": natural_order_violation_row_count > 0,
        "natural_order_violation_row_count": natural_order_violation_row_count,
        "parser_status": trajectory.parser_status,
    }


def verify_image(
    *,
    image_id: int,
    payload: Path | None,
    output_dir: Path,
    repetition_penalty: float,
    source_only: bool,
    stage: str = "N2",
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    spec = _stage_spec(stage)
    if image_id not in spec.image_ids:
        raise HoldError(f"verify image must be one of {spec.image_ids} for {stage}")
    if source_only != (payload is None):
        raise HoldError("source-only requires no payload; candidate verify requires payload")
    check_bindings(load_tokenizer=False)
    panel = {int(row["image_id"]): row for row in load_panel()}
    started = time.perf_counter()
    context, session, (_example, request), runtime_identity, components = _runtime_setup(image_id)
    try:
        route_ids, route_receipt = tokenize_canonical_route(components.tokenizer, panel[image_id])
        payload_sha: str | None = None
        hook_context: Any = nullcontext()
        payload_receipt: dict[str, Any] | None = None
        pre_source_rp1_0: Mapping[str, Any] | None = None
        if payload is not None:
            payload_receipt = json.loads(payload.with_suffix(".json").read_text())
            payload_sha = sha256_file(payload)
            if payload_receipt.get("payload_sha256") != payload_sha:
                raise HoldError("payload hash differs from sidecar receipt")
            if (
                payload_receipt.get("stage") != stage
                or tuple(payload_receipt.get("image_ids", ())) != spec.image_ids
                or int(payload_receipt.get("owner_count", -1)) != spec.owner_count
            ):
                raise HoldError("payload stage/image receipt binding differs")
            payload_captures = payload_receipt.get("captures")
            if (
                not isinstance(payload_captures, list)
                or len(payload_captures) != len(spec.image_ids)
                or any(not isinstance(item, Mapping) for item in payload_captures)
                or tuple(int(item.get("image_id", -1)) for item in payload_captures)
                != spec.image_ids
            ):
                raise HoldError("payload capture order differs from stage image set")
            pre_source_rp1_0 = next(
                item["pre_source_rp1_0"]
                for item in payload_captures
                if int(item["image_id"]) == image_id
            )
            tensors = load_file(str(payload))
            hook_context = SelectedOutputRowsHook(
                _output_head(components.model),
                tensors["selected_token_ids"],
                tensors["residual_rows"],
            )
        with hook_context:
            result = _decode(session, request, repetition_penalty=repetition_penalty)
        evaluation = _evaluate_result(
            result=result,
            image_id=image_id,
            canonical_ids=route_ids,
            backend_version=str(session.receipt.backend_version),
            repetition_penalty=repetition_penalty,
        )
        mode = "source" if source_only else "candidate"
        suffix = str(repetition_penalty).replace(".", "p")
        path = output_dir / f"verify-{image_id}-{mode}-rp{suffix}.json"
        receipt = {
            "schema_version": "human13_output_qp_verify.v1",
            "status": "verified_natural_greedy",
            "mode": mode,
            "stage": stage,
            "image_ids": list(spec.image_ids),
            "image_id": image_id,
            "repetition_penalty": repetition_penalty,
            "payload_sha256": payload_sha,
            "pre_source_rp1_0": pre_source_rp1_0,
            "route": route_receipt,
            "evaluation": evaluation,
            "runtime_identity": runtime_identity,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
            "process_id": os.getpid(),
            "model_load_count": 1,
            "natural_generation_count": 1,
        }
        immutable_json(path, receipt)
        return {**receipt, "receipt_path": str(path.resolve())}
    finally:
        context.__exit__(None, None, None)


def aggregate_results(
    *, results: Sequence[Path], post_source: Sequence[Path], stage: str = "N2"
) -> dict[str, Any]:
    spec = _stage_spec(stage)
    if len(results) != len(spec.image_ids) or len(post_source) != len(spec.image_ids):
        raise HoldError(
            f"aggregate {stage} requires {len(spec.image_ids)} candidates and "
            f"{len(spec.image_ids)} post-Source receipts"
        )
    candidates = [json.loads(path.read_text()) for path in results]
    post = [json.loads(path.read_text()) for path in post_source]
    if tuple(int(item.get("image_id", -1)) for item in candidates) != spec.image_ids:
        raise HoldError(f"candidate result order differs from frozen {stage} order")
    if tuple(int(item.get("image_id", -1)) for item in post) != spec.image_ids:
        raise HoldError(f"post-Source result order differs from frozen {stage} order")
    if any(
        item.get("stage") != stage
        or tuple(item.get("image_ids", ())) != spec.image_ids
        for item in (*candidates, *post)
    ):
        raise HoldError("aggregate receipt stage/image binding differs")
    if any(
        item.get("mode") != "candidate" or item.get("repetition_penalty") != 1.0
        for item in candidates
    ):
        raise HoldError(f"{stage} primary aggregation requires RP1.0 candidate receipts")
    payloads = {item.get("payload_sha256") for item in candidates}
    if len(payloads) != 1 or None in payloads:
        raise HoldError("candidate results do not share one payload")
    source_restored = True
    for candidate, restored in zip(candidates, post, strict=True):
        if restored.get("mode") != "source" or restored.get("repetition_penalty") != 1.0:
            raise HoldError("post-Source receipt is not RP1.0 source-only")
        pre = candidate.get("pre_source_rp1_0")
        observed = restored.get("evaluation", {})
        if not isinstance(pre, Mapping) or any(
            pre.get(field) != observed.get(field)
            for field in ("generated_token_ids_sha256", "parser_text_sha256", "stop_reason")
        ):
            source_restored = False
    evaluations = [item["evaluation"] for item in candidates]
    owner_iou50 = sum(int(item["matched_owner_count"]["50"]) for item in evaluations)
    hard_debt = sum(
        int(item["duplicate_count"])
        + int(item["unmatched_prediction_count"])
        + int(item["malformed_count"])
        + int(bool(item["cap_debt"]))
        for item in evaluations
    )
    passed = (
        owner_iou50 == spec.owner_count
        and hard_debt == 0
        and all(bool(item["natural_eos"]) for item in evaluations)
        and source_restored
    )
    receipt = {
        "schema_version": "human13_output_qp_aggregate.v1",
        "status": "passed" if passed else "HOLD",
        "stage": stage,
        "image_ids": list(spec.image_ids),
        "payload_sha256": next(iter(payloads)),
        "owner_counts": {
            threshold: sum(int(item["matched_owner_count"][threshold]) for item in evaluations)
            for threshold in ("50", "60", "80")
        },
        "required_owner_count": spec.owner_count,
        "hard_debt": hard_debt,
        "all_natural_eos": all(bool(item["natural_eos"]) for item in evaluations),
        "source_a_b_a_restored": source_restored,
        "exact_route_diagnostic_count": sum(bool(item["exact_route"]) for item in evaluations),
    }
    print(json.dumps(receipt, sort_keys=True))
    if not passed:
        raise HoldError(f"{stage} primary gate failed")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-bindings", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    capture = subparsers.add_parser("capture")
    capture.add_argument("--stage", choices=tuple(STAGES), default="N2")
    capture.add_argument("--image-id", type=int, required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    solve = subparsers.add_parser("solve")
    solve.add_argument("--stage", choices=tuple(STAGES), default="N2")
    solve.add_argument("--capture", type=Path, nargs="+", required=True)
    solve.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--stage", choices=tuple(STAGES), default="N2")
    verify.add_argument("--image-id", type=int, required=True)
    verify.add_argument("--payload", type=Path)
    verify.add_argument("--source-only", action="store_true")
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.add_argument("--rp", type=float, choices=(1.0, 1.10), default=1.0)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--stage", choices=tuple(STAGES), default="N2")
    aggregate.add_argument("--results", type=Path, nargs="+", required=True)
    aggregate.add_argument("--post-source", type=Path, nargs="+", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.check_bindings:
            if args.command is not None:
                raise HoldError("--check-bindings cannot be combined with a command")
            result = check_bindings()
        elif args.command == "capture":
            result = capture_image(
                image_id=args.image_id, output_dir=args.output_dir, stage=args.stage
            )
        elif args.command == "solve":
            result = solve_captures(
                captures=args.capture, output_dir=args.output_dir, stage=args.stage
            )
        elif args.command == "verify":
            result = verify_image(
                image_id=args.image_id,
                payload=args.payload,
                output_dir=args.output_dir,
                repetition_penalty=args.rp,
                source_only=args.source_only,
                stage=args.stage,
            )
        elif args.command == "aggregate":
            aggregate_results(
                results=args.results, post_source=args.post_source, stage=args.stage
            )
            return 0
        else:
            _parser().print_help()
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0
    except (HoldError, FileExistsError, ValueError) as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
