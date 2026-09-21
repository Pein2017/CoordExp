"""CPU-only checks for Lane A prefix, row, and accounting contracts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import probes.training_set_completion.spatial_progress_gate.runtime as runtime
from probes.training_set_completion.spatial_progress_gate.runtime import (
    ADMISSION,
    _cells,
    _condition_prefix,
    _falsification,
    _load_admission,
    _actual_row,
    _actual_tokens,
    _resolve_output,
    _source_record,
    _validate_row,
)


def run_selfcheck() -> None:
    admission, _ = _load_admission()
    cells = _cells(admission)
    assert len(cells) == 8
    for boundary in admission["lane_a"]["boundaries"]:
        native = [151646, 2190, 151647, 151648, 152458, 152340, 152520, 152367, 151649]
        prefix_end = int(boundary["prefix_end"])
        native = [0] * prefix_end + native
        native[int(boundary["conditions"][1]["absolute_offset"])] = int(boundary["conditions"][1]["old_token_id"])
        before, before_spec = _condition_prefix(boundary, native, "x1_before_N786")
        assert before[before_spec["absolute_offset"]] == before_spec["new_token_id"]
        assert before_spec["changed_prefix_check"]["passed"]
        assert before[before_spec["absolute_offset"]] < 151670 + 786
        after, after_spec = _condition_prefix(boundary, native, "native")
        assert after == native[:prefix_end]
        assert after_spec["changed_prefix_check"]["passed"]

    tokens = [151646, 2190, 151647, 151648, 152454, 152340, 152521, 152374, 151649]
    row = {
        "token_ids": tokens,
        "token_logprobs": [-1.0] * len(tokens),
        "row_sum_logprob": -float(len(tokens)),
        "positions": [[0, i, i] for i in range(len(tokens))],
        "token_roles": ["entry", "description", "description_end", "box_start", "x1", "y1", "x2", "y2", "terminator"],
        "conditional_xy1": {
            "path": "x1_then_y1",
            "token_ids": [tokens[4], tokens[5]],
            "token_logprobs": [-1.0, -1.0],
            "logprob_sum": -2.0,
        },
    }
    _validate_row(row)
    falsification = _falsification({"rows": {"synthetic": row}})
    assert falsification["passed"] and all(item["rejected"] for item in falsification["checks"])
    corrupted = copy.deepcopy(row)
    corrupted["token_ids"] = corrupted["token_ids"][:-1]
    try:
        _validate_row(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("terminator mutation was accepted")

    # Exercise the caller boundary that consumes parser metadata.  The scorer
    # and trace comparator are inert CPU stubs here, so this check performs no
    # model or vision forwards while proving entry and terminator coverage.
    cell = next(item for item in cells if item["id"] == "tied-14038-failure-before-row8--x1_before_N786")
    record = _source_record(admission, cell["source_boundary_id"])
    native = [
        int(token)
        for token in json.loads(Path(record["raw"]["path"]).read_text())["rows"][int(record["batch_index"])]
        ["token_ids"]
    ]
    actual = _actual_row(native, next(row["index"] for row in runtime.parsed_rows(native) if int(row["start"]) == int(cell["prefix_end"])))
    assert "tokens" not in actual
    actual_tokens = _actual_tokens(native, actual)
    assert actual_tokens == native[int(actual["start"]):int(actual["end"])]
    calls = {"score": 0}
    old_score = runtime._score_candidate
    old_compare = runtime._trace_compare

    def fake_score(*_args, **_kwargs):
        calls["score"] += 1
        return {"boundary_logits": None, "action_logits": [None] * len(actual_tokens)}

    def fake_compare(*, logits, trace, batch_index, absolute_offset, token_id, role):
        return {"passed": True, "logprob_abs_error": 0.0, "top2_max_abs_error": 0.0, "role": role}

    runtime._score_candidate = fake_score
    runtime._trace_compare = fake_compare
    try:
        parity = runtime._trace_parity(
            model=object(), batch=None, raw=[], target=0, native=native,
            prefix_end=int(cell["prefix_end"]), trace={}, pad=0, device=torch.device("cpu"),
        )
    finally:
        runtime._score_candidate = old_score
        runtime._trace_compare = old_compare
    assert parity["passed"] and parity["entry_included"] and parity["terminator_included"]
    assert len(parity["tokens"]) == len(actual_tokens) + 1 and calls["score"] == 1

    explicit = runtime.ROOT / "diagnostics" / "explicit-output-probe"
    assert not explicit.exists()
    resolved = _resolve_output(SimpleNamespace(
        output=explicit, mode="qualify", repair=True, retry=0,
    ), cell)
    assert resolved == explicit.resolve()
    assert runtime.PREDECESSOR_ROOT != runtime.ROOT
    assert runtime.ADMISSION.parent.parent == runtime.PREDECESSOR_ROOT

    assert json.loads(ADMISSION.read_text())["status"] == "frozen_before_intervention_scores"
    print("spatial_progress_gate selfcheck: PASS frozen cells, crossing prefix, complete rows, x1/y1 path, falsification, caller trace boundary")


if __name__ == "__main__":
    run_selfcheck()
