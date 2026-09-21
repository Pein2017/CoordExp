#!/usr/bin/env python3
"""CPU-only GREEN witnesses for the corrected Lane-B helpers."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
ARTIFACT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-19-recurrence-spatial-source/final/correction"
)
sys.path.insert(0, str(WORKTREE))

from probes.training_set_completion.recurrence_spatial import prepare, producer, reduce  # noqa: E402
from probes.training_set_completion.recurrence_spatial.recurrence_semantics import complete_box_count, parse_rows, runs  # noqa: E402
from probes.training_set_completion.numerical_feedback.select import choose_episode, rows as accepted_rows  # noqa: E402


class DummyTokenizer:
    def decode(self, tokens, **kwargs):
        return "same"


def logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def main() -> None:
    source_width, source_height, canvas_width, canvas_height, tx = 1024, 768, 1280, 768, 128
    coord = prepare.COORD_BASE
    tokens = [prepare.OBJ_START, 42, prepare.OBJ_END, prepare.BOX_START, coord + 100, coord + 200, coord + 300, coord + 400, prepare.BOX_END]
    transformed, boxes = prepare.transform_history(
        tokens,
        source_width=source_width,
        source_height=source_height,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        tx=tx,
    )
    assert boxes[0]["mapped_bins"][1] == 200 and boxes[0]["mapped_bins"][3] == 400

    geometry = {
        "source_width": source_width,
        "source_height": source_height,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
    }
    cell = {"visual_offset_px": tx}
    producer_inverse = [producer._inverse_output_bin(value, cell=cell, geometry=geometry, coordinate_index=index) for index, value in enumerate([100, 200, 300, 400])]
    reducer_inverse = [reduce.inverse(value, source_width=source_width, source_height=source_height, canvas_width=canvas_width, canvas_height=canvas_height, tx=tx, coordinate_index=index) for index, value in enumerate([100, 200, 300, 400])]
    assert producer_inverse[1] == 200 and producer_inverse[3] == 400
    assert reducer_inverse[1] == 200 and reducer_inverse[3] == 400

    input_ids = torch.tensor([[producer.OBJ_END] * 32 + [producer.BOX_END] * 31], dtype=torch.long)
    scores = torch.zeros((1, 152700), dtype=torch.float32)
    not_ready = producer.RowLimitEOS(baseline_end_count=0, max_rows=32, eos=producer.EOS)(input_ids, scores)
    assert not bool(torch.isneginf(not_ready[0, 0]))
    # Stray BOX_END tokens do not consume a serialized row budget.
    stray_ids = torch.tensor([[producer.BOX_END] * 32], dtype=torch.long)
    stray = producer.RowLimitEOS(baseline_end_count=0, max_rows=32, eos=producer.EOS)(stray_ids, scores)
    assert complete_box_count(stray_ids[0].tolist()) == 0
    assert not bool(torch.isneginf(stray[0, 0]))
    # Invalid geometry still consumes a serialized complete row.
    invalid_row = [producer.OBJ_START, 42, producer.OBJ_END, producer.BOX_START, producer.COORD_BASE + 200, producer.COORD_BASE + 200, producer.COORD_BASE + 100, producer.COORD_BASE + 100, producer.BOX_END]
    ready_ids = torch.tensor([invalid_row * 32], dtype=torch.long)
    ready = producer.RowLimitEOS(baseline_end_count=0, max_rows=32, eos=producer.EOS)(ready_ids, scores)
    assert bool(torch.isneginf(ready[0, 0]))
    # The row cap is measured on the generated suffix, with explicit
    # provenance for an injected EOS.  Prompt rows do not spend the free
    # continuation budget.
    prefix = invalid_row
    suffix_ids = torch.tensor([prefix + invalid_row * 32], dtype=torch.long)
    suffix_limit = producer.RowLimitEOS(
        baseline_input_width=len(prefix),
        baseline_complete_rows=1,
        max_rows=32,
        eos=producer.EOS,
    )
    suffix_ready = suffix_limit(suffix_ids, scores)
    assert bool(torch.isneginf(suffix_ready[0, 0]))
    suffix_receipt = suffix_limit.receipt()
    assert suffix_receipt["free_suffix_start"] == len(prefix)
    assert suffix_receipt["baseline_complete_rows"] == 1
    assert suffix_receipt["injected_eos"] and suffix_receipt["injection_reason"] == "row_cap"

    # Exercise the actual _run_free wrapper/caller return path with a fake
    # native generate entry.  This catches an unreachable receipt return while
    # keeping the witness CPU-only and model-free.
    class FakeGenerateModel:
        def __init__(self):
            self.saw_injected_eos = False

        def generate(self, **kwargs):
            processors = list(kwargs["logits_processor"])
            prompt = kwargs["input_ids"]
            scores = torch.zeros((1, 152700), dtype=torch.float32)
            for processor in processors:
                scores = processor(prompt, scores)
            extended = torch.tensor([prefix + invalid_row * 32], dtype=torch.long)
            extended_scores = torch.zeros((1, 152700), dtype=torch.float32)
            for processor in processors:
                extended_scores = processor(extended, extended_scores)
            self.saw_injected_eos = bool(torch.isneginf(extended_scores[0, 0]))
            return extended

    fake_model = FakeGenerateModel()
    original_continuations = producer.generate_continuations

    def fake_continuations(model, *args, **kwargs):
        model.generate(
            input_ids=torch.tensor([prefix], dtype=torch.long),
            logits_processor=[],
        )
        return [SimpleNamespace(token_ids=[producer.EOS], stop_reason="eos", policy_logprobs=[])]

    producer.generate_continuations = fake_continuations
    try:
        fake_result, fake_receipt = producer._run_free(
            fake_model,
            batch=None,
            history=[],
            eos=producer.EOS,
            pad=producer.EOS,
        )
    finally:
        producer.generate_continuations = original_continuations
    assert fake_result.stop_reason == "eos"
    assert fake_model.saw_injected_eos
    assert fake_receipt["injected_eos"] and fake_receipt["injection_reason"] == "row_cap"

    chain = [
        {"row_index": 0, "status": "valid", "complete": True, "source_geometry_valid": True, "description": "same", "description_tokens": [42], "coord_bins_source": [100, 100, 200, 200]},
        {"row_index": 1, "status": "valid", "complete": True, "source_geometry_valid": True, "description": "same", "description_tokens": [42], "coord_bins_source": [105, 100, 205, 200]},
        {"row_index": 2, "status": "valid", "complete": True, "source_geometry_valid": True, "description": "same", "description_tokens": [42], "coord_bins_source": [110, 100, 210, 200]},
    ]
    invalid = [
        {"row_index": 0, "status": "invalid", "complete": True, "source_geometry_valid": False, "description": "same", "description_tokens": [42], "coord_bins_source": [200, 200, 100, 100]},
        {"row_index": 1, "status": "valid", "complete": True, "source_geometry_valid": True, "description": "same", "description_tokens": [42], "coord_bins_source": [201, 200, 101, 100]},
        {"row_index": 2, "status": "valid", "complete": True, "source_geometry_valid": True, "description": "same", "description_tokens": [42], "coord_bins_source": [202, 200, 102, 100]},
    ]
    assert not runs(chain, near=True)
    assert len(runs(invalid, near=True)) == 1
    assert len(producer._runs(invalid, near=True)) == 1
    # A,B,A,B,A is not an accepted consecutive triple, even though a
    # nonconsecutive all-pairs clique implementation would find three A rows.
    ababa = []
    for index, description_token in enumerate([42, 43, 42, 43, 42]):
        ababa.append(
            {
                "row_index": index,
                "status": "valid",
                "complete": True,
                "source_geometry_valid": True,
                "description": str(description_token),
                "description_tokens": [description_token],
                "coord_bins_source": [100 + index, 100, 200 + index, 200],
            }
        )
    assert runs(ababa, near=True) == []
    accepted_ababa = [
        [151646, token, 151647, 151648, 151670 + 100 + index, 151670 + 100, 151670 + 200 + index, 151670 + 200, 151649]
        for index, token in enumerate([42, 43, 42, 43, 42])
    ]
    accepted_ababa_flat = [token for row in accepted_ababa for token in row]
    assert choose_episode(accepted_rows(accepted_ababa_flat)) == (None, None)

    # Four coordinates must be contiguous immediately before BOX_END;
    # unrelated tokens inside the coordinate segment are malformed.
    malformed = [
        producer.OBJ_START,
        42,
        producer.OBJ_END,
        producer.BOX_START,
        producer.COORD_BASE + 100,
        999,
        producer.COORD_BASE + 200,
        producer.COORD_BASE + 300,
        producer.COORD_BASE + 400,
        producer.BOX_END,
    ]
    parsed_malformed = parse_rows(
        malformed,
        DummyTokenizer(),
        cell=cell,
        geometry=geometry,
    )
    assert parsed_malformed["complete_rows"] == 0
    assert parsed_malformed["malformed_rows"] == 1

    logits = [-abs(index - 600.0) for index in range(1000)]
    log_probs = [-abs(index - 600.0) - 8.0 for index in range(1000)]
    window = producer._window_measurement(
        {"coordinate_logits": logits, "coordinate_log_probs": log_probs},
        old_bin=500,
        moved_bin=700,
    )
    assert window["window_radius"] == 8
    assert window["old_window"]["lo"] == 492 and window["old_window"]["hi"] == 508
    assert window["moved_window"]["lo"] == 692 and window["moved_window"]["hi"] == 708
    assert math.isclose(window["moved_minus_old_log_mass"], 0.0, abs_tol=1e-12)

    green = {
        "schema": "recurrence_spatial_source.correction_green_witness.v1",
        "status": "green_witnesses_passed",
        "no_model_loaded": True,
        "no_gpu_forwards": True,
        "witnesses": {
            "history_role_mapping": {"mapped_bins": boxes[0]["mapped_bins"], "y_unchanged": True},
            "producer_inverse_role_mapping": {"bins": producer_inverse, "y_unchanged": True},
            "reducer_inverse_role_mapping": {"bins": reducer_inverse, "y_unchanged": True},
            "full_box_row_stop": {
                "description_end_32_box_end_31_not_ready": True,
                "stray_box_end_32_not_ready": True,
                "box_end_32_ready_even_if_invalid": True,
                "free_suffix_boundary_and_injected_eos": suffix_receipt,
                "run_free_wrapper_return_and_provenance": fake_receipt,
                "extraneous_coordinate_token_rejected": True,
            },
            "accepted_consecutive_triple": {
                "adjacent_chain_rejected": True,
                "invalid_complete_rows_retained": True,
                "ababa_nonconsecutive_falsification": True,
                "accepted_selector_result": [None, None],
            },
            "forced_window": {"radius": 8, "old_window": [492, 508], "moved_window": [692, 708], "full_vocab_log_mass": True},
        },
    }
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT / "green-witness.json").write_text(json.dumps(green, indent=2) + "\n")
    print(json.dumps(green, indent=2))


if __name__ == "__main__":
    main()
