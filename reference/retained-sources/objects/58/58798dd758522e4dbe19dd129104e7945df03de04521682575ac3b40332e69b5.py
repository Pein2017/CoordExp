#!/usr/bin/env python3
"""Executable pre-correction RED witnesses for the Lane-B producer/reducer.

This file is intentionally kept beside the immutable candidate-v1 snapshot.
It imports the worktree helpers as they existed before the scoped correction,
and records counterexamples without loading a model or touching a GPU.
"""

from __future__ import annotations

import json
import math
import sys
import importlib.util
from pathlib import Path

import torch


WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
SNAPSHOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-19-recurrence-spatial-source-candidate-v1/code_snapshot"
)
ARTIFACT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-19-recurrence-spatial-source/final/correction"
)
sys.path.insert(0, str(WORKTREE))


def _load_snapshot_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load snapshot module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Import the exact pre-correction code saved by candidate-v1.  The producer's
# absolute prepare import is rebound to the snapshot module for this process.
prepare = _load_snapshot_module(
    "probes.training_set_completion.recurrence_spatial.prepare",
    SNAPSHOT / "prepare.py",
)
producer = _load_snapshot_module(
    "probes.training_set_completion.recurrence_spatial.producer_candidate_v1",
    SNAPSHOT / "producer.py",
)
reduce = _load_snapshot_module(
    "probes.training_set_completion.recurrence_spatial.reduce_candidate_v1",
    SNAPSHOT / "reduce.py",
)


class DummyTokenizer:
    def decode(self, tokens, **kwargs):
        return "same"


def role_expected(value: int, role: str, *, source_width: int, source_height: int, canvas_width: int, canvas_height: int, tx: int, ty: int) -> int:
    if role in {"x1", "x2"}:
        return prepare.map_bin(value, source_width=source_width, canvas_width=canvas_width, tx=tx)
    # A horizontal transform on a common canvas leaves y in the same pixel
    # domain.  This is deliberately an independent expected implementation.
    source_pixel = value * (source_height - 1) / 999.0
    canvas_pixel = source_pixel + ty
    return int(round(canvas_pixel * 999.0 / (canvas_height - 1)))


def all_pair_triple(rows: list[dict], *, eps: int) -> bool:
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            for k in range(j + 1, len(rows)):
                trio = (rows[i], rows[j], rows[k])
                if len({row.get("description") for row in trio}) != 1:
                    continue
                if all(
                    max(abs(a - b) for a, b in zip(left["coord_bins_source"], right["coord_bins_source"], strict=True)) <= eps
                    for left, right in ((rows[i], rows[j]), (rows[i], rows[k]), (rows[j], rows[k]))
                ):
                    return True
    return False


def naive_nonconsecutive_clique(rows: list[dict], *, eps: int) -> bool:
    """Reproduce the pre-gate all-pairs candidate-membership mistake."""

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            for k in range(j + 1, len(rows)):
                trio = (rows[i], rows[j], rows[k])
                if len({row.get("description") for row in trio}) != 1:
                    continue
                if all(
                    max(abs(a - b) for a, b in zip(left["coord_bins_source"], right["coord_bins_source"], strict=True)) <= eps
                    for left, right in ((rows[i], rows[j]), (rows[i], rows[k]), (rows[j], rows[k]))
                ):
                    return True
    return False


def corrected_window(log_probs: list[float], logit: list[float], *, old_bin: int, moved_bin: int, radius: int = 8) -> dict:
    def one(center: int, values: list[float]) -> dict:
        lo, hi = max(0, center - radius), min(999, center + radius)
        selected = values[lo : hi + 1]
        max_value = max(selected)
        return {"lo": lo, "hi": hi, "log_mass": _logsumexp(selected), "max_log_prob": max_value}

    old = one(old_bin, log_probs)
    moved = one(moved_bin, log_probs)
    return {
        "radius": radius,
        "old_bin": old_bin,
        "moved_bin": moved_bin,
        "old": old,
        "moved": moved,
        "moved_minus_old_log_mass": moved["log_mass"] - old["log_mass"],
        "old_single_log_prob": log_probs[old_bin],
        "moved_single_log_prob": log_probs[moved_bin],
        "logit_window_max_delta": moved["max_log_prob"] - old["max_log_prob"],
        "_logit_reference": logit[old_bin],
    }


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def main() -> None:
    coord_base = prepare.COORD_BASE
    token_ids = [
        prepare.OBJ_START,
        42,
        prepare.OBJ_END if hasattr(prepare, "OBJ_END") else 151647,
        prepare.BOX_START,
        coord_base + 100,
        coord_base + 200,
        coord_base + 300,
        coord_base + 400,
        prepare.BOX_END,
    ]
    source_width, source_height, canvas_width, canvas_height, tx, ty = 1024, 768, 1280, 768, 128, 0
    transformed, boxes = prepare.transform_history(
        token_ids,
        source_width=source_width,
        canvas_width=canvas_width,
        tx=tx,
    )
    source_bins = [100, 200, 300, 400]
    expected_bins = [
        role_expected(value, role, source_width=source_width, source_height=source_height, canvas_width=canvas_width, canvas_height=canvas_height, tx=tx, ty=ty)
        for value, role in zip(source_bins, ("x1", "y1", "x2", "y2"), strict=True)
    ]

    geometry = {
        "source_width": source_width,
        "source_height": source_height,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
    }
    cell = {"visual_offset_px": tx}
    inverse_expected = [
        value if role in {"y1", "y2"} else prepare.inverse_bin(value, source_width=source_width, canvas_width=canvas_width, tx=tx)
        for value, role in zip([100, 200, 300, 400], ("x1", "y1", "x2", "y2"), strict=True)
    ]
    producer_inverse = [producer._inverse_output_bin(value, cell=cell, geometry=geometry) for value in [100, 200, 300, 400]]
    reduce_inverse = [reduce.inverse(value, source_width=source_width, canvas_width=canvas_width, tx=tx) for value in [100, 200, 300, 400]]

    # 32 description terminators but only 31 complete BOX_END terminators.
    input_ids = torch.tensor([[producer.OBJ_END] * 32 + [producer.BOX_END] * 31], dtype=torch.long)
    scores = torch.zeros((1, 152700), dtype=torch.float32)
    limited = producer.RowLimitEOS(baseline_end_count=0, max_rows=32, eos=producer.EOS)(input_ids, scores)
    row_cap_forced = bool(torch.isneginf(limited[0, 0]))
    stray_box_end_tokens = [producer.BOX_END] * 32
    naive_box_end_count = stray_box_end_tokens.count(producer.BOX_END)

    chain = [
        {"row_index": 0, "status": "valid", "source_geometry_valid": True, "description": "same", "coord_bins_source": [100, 100, 200, 200]},
        {"row_index": 1, "status": "valid", "source_geometry_valid": True, "description": "same", "coord_bins_source": [105, 100, 205, 200]},
        {"row_index": 2, "status": "valid", "source_geometry_valid": True, "description": "same", "coord_bins_source": [110, 100, 210, 200]},
    ]
    invalid = [
        {"row_index": 0, "status": "invalid", "source_geometry_valid": False, "description": "same", "coord_bins_source": [200, 200, 100, 100]},
        {"row_index": 1, "status": "valid", "source_geometry_valid": True, "description": "same", "coord_bins_source": [201, 200, 101, 100]},
        {"row_index": 2, "status": "valid", "source_geometry_valid": True, "description": "same", "coord_bins_source": [202, 200, 102, 100]},
    ]
    current_chain = producer._runs(chain, near=True)
    current_invalid = producer._runs(invalid, near=True)
    ababa = [
        {
            "row_index": index,
            "status": "valid",
            "description": "A" if index % 2 == 0 else "B",
            "coord_bins_source": [100 + index, 100, 200 + index, 200],
        }
        for index in range(5)
    ]

    logits = [-abs(index - 600.0) for index in range(1000)]
    log_probs = [-abs(index - 600.0) - 8.0 for index in range(1000)]
    scalar_window = producer._window_measurement(
        {"coordinate_logits": logits, "coordinate_log_probs": log_probs},
        old_bin=500,
        moved_bin=700,
    )
    fixed_window = corrected_window(log_probs, logits, old_bin=500, moved_bin=700)

    witness = {
        "schema": "recurrence_spatial_source.correction_red_witness.v1",
        "status": "red_witnesses_observed",
        "candidate_snapshot": "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source-candidate-v1",
        "witnesses": {
            "history_role_mapping": {
                "observed_mapped_bins": boxes[0]["mapped_bins"],
                "role_aware_expected_bins": expected_bins,
                "current_y_changed": boxes[0]["mapped_bins"][1] != expected_bins[1] or boxes[0]["mapped_bins"][3] != expected_bins[3],
            },
            "producer_inverse_role_mapping": {
                "current": producer_inverse,
                "role_aware_expected": inverse_expected,
                "current_y_changed": producer_inverse[1] != inverse_expected[1] or producer_inverse[3] != inverse_expected[3],
            },
            "reducer_inverse_role_mapping": {
                "current": reduce_inverse,
                "role_aware_expected": inverse_expected,
                "current_y_changed": reduce_inverse[1] != inverse_expected[1] or reduce_inverse[3] != inverse_expected[3],
            },
            "full_box_row_stop": {
                "description_end_count": 32,
                "box_end_count": 31,
                "current_forced_eos": row_cap_forced,
                "correct_full_box_row_cap_forced": False,
                "stray_box_end_tokens": 32,
                "naive_box_end_counter_would_force": naive_box_end_count >= 32,
                "serialized_complete_box_count": 0,
            },
            "pairwise_near": {
                "adjacent_chain_current_runs": current_chain,
                "all_pair_triple_expected": all_pair_triple(chain, eps=8),
                "invalid_rows_current_runs": current_invalid,
                "invalid_rows_all_pair_triple_expected": all_pair_triple(invalid, eps=8),
                "nonconsecutive_ababa_naive_clique": naive_nonconsecutive_clique(ababa, eps=8),
                "nonconsecutive_ababa_accepted_consecutive_triple_expected": False,
            },
            "forced_window": {
                "current_single_bin_measurement": scalar_window,
                "fixed_plus_minus_8_measurement": fixed_window,
                "current_has_window_radius": "window_radius" in scalar_window,
            },
        },
    }
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT / "red-witness.json").write_text(json.dumps(witness, indent=2) + "\n")
    print(json.dumps(witness, indent=2))


if __name__ == "__main__":
    main()
