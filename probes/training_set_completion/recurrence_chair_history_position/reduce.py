"""Cold reduction of the frozen chair y1 grid; makes no model calls."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.recurrence_first_arrivals.prepare import _require
from probes.training_set_completion.recurrence_first_arrivals.stage1_case import _write_new
from probes.training_set_completion.recurrence_chair_history_position.run import MANIFEST, OUTPUT, PROTOCOL, readback


def reduce(out: Path) -> dict:
    _require(readback(out)["status"] == "passed", "cold grid readback failed")
    m = json.loads(MANIFEST.read_text())
    pilot = json.loads((out / "pilot.json").read_text())
    terminal = json.loads((out / "receipt.json").read_text())
    pre = json.loads((out / "preflight.json").read_text())
    records, consumers, vectors = {}, {}, {}
    for entry in pilot["completed"]:
        record = json.loads(Path(entry["record"]["path"]).read_text())
        key = (record["mode"], record["candidate_owner"], record["history"], record["positions"])
        _require(key not in records, "duplicate logical cell")
        records[key] = record
        consumers[key] = json.loads(Path(record["consumer"]["path"]).read_text())
        vectors[key] = torch.load(record["vector"]["path"], map_location="cpu", weights_only=True)
    _require(len(records) == 18, "cold grid denominator changed")
    checks = []
    for probe in m["x1_probes"]:
        owner = probe["candidate_owner"]
        for history in ("E", "L_A", "L_B"):
            a = consumers["y1_grid", owner, history, "E"]
            b = consumers["y1_grid", owner, history, "L"]
            _require(a["input_ids"] == b["input_ids"] and a["attention_mask"] == b["attention_mask"] and
                     a["attention_mask_hashes"] == b["attention_mask_hashes"] and
                     a["cache_position_hashes"] == b["cache_position_hashes"],
                     "fixed-history physical input/mask/cache order changed")
            expected = list(range(len(a["input_ids"][0]) - 5, len(a["input_ids"][0])))
            differences = [[i for i, (x, y) in enumerate(zip(a["position_ids"][axis][0],
                                                             b["position_ids"][axis][0], strict=True)) if x != y]
                           for axis in range(3)]
            _require(differences == [expected] * 3 and a["position_ids"] == a["rotary_position_ids"] and
                     b["position_ids"] == b["rotary_position_ids"], "S-only MRoPE crossing changed")
            checks.append({"candidate_owner": owner, "history": history,
                           "S_only_position_difference_indices": expected,
                           "same_physical_input_mask_cache_order": True})
        for position in ("E", "L"):
            a = consumers["y1_grid", owner, "L_A", position]
            b = consumers["y1_grid", owner, "L_B", position]
            differences = [i for i, (x, y) in enumerate(zip(a["input_ids"][0], b["input_ids"][0], strict=True))
                           if x != y]
            expected = [pre["shape"]["prompt_width"] + m["early_row_start"] + i for i in (5, 6)]
            _require(differences == expected and a["position_ids"] == b["position_ids"] and
                     a["attention_mask"] == b["attention_mask"] and
                     a["cache_position_hashes"] == b["cache_position_hashes"],
                     "A-to-B content changed other than historical y1/x2")
            checks.append({"candidate_owner": owner, "positions": position,
                           "A_to_B_only_physical_token_indices": differences,
                           "same_positions_mask_cache_order": True})
    results = []
    for probe in m["x1_probes"]:
        owner = probe["candidate_owner"]
        e, l = (151670 + probe[f"reference_{side}_y1_bin"] for side in ("E", "L"))
        matrix, margins, winners = [], {}, {}
        for history in ("E", "L_A", "L_B"):
            for position in ("E", "L"):
                key = ("y1_grid", owner, history, position)
                record, vector = records[key], vectors[key]
                z_e, z_l = float(vector[e]), float(vector[l])
                margins[history, position] = z_e - z_l
                winners[history, position] = record["top2_ids"][0]
                matrix.append({"history": history, "positions": position,
                               "winner_token": record["top2_ids"][0],
                               "winner_y1_bin": record["top2_ids"][0] - 151670,
                               "runner_token": record["top2_ids"][1],
                               "top2_gap": record["top2_gap"],
                               "near_tie": record["top2_gap"] <= .001,
                               "logsumexp": record["logsumexp"],
                               "z_early": z_e, "z_late": z_l,
                               "m_early_minus_late": z_e - z_l,
                               "vector": record["vector"], "consumer": record["consumer"]})
        pos = {h: margins[h, "L"] - margins[h, "E"] for h in ("E", "L_A", "L_B")}
        history = {p: margins["L_A", p] - margins["E", p] for p in ("E", "L")}
        content = {p: margins["L_B", p] - margins["L_A", p] for p in ("E", "L")}
        results.append({"candidate_owner": owner, "early_y1_token": e, "late_y1_token": l,
                        "matrix": matrix,
                        "contrasts": {"position_L_minus_E_by_history": pos,
                                      "added_history_LA_minus_E_by_position": history,
                                      "content_LB_minus_LA_by_position": content,
                                      "history_by_position_interaction": pos["L_A"] - pos["E"],
                                      "content_by_position_interaction": pos["L_B"] - pos["L_A"]},
                        "prediction_checks": {
                            "strict_position_transport": winners["E", "L"] == l and winners["L_A", "E"] == e,
                            "strict_history_package_following": winners["E", "L"] == e and winners["L_A", "E"] == l,
                            "strict_B_content_restoration_at_L": winners["L_B", "L"] == e}})
    source_errors, reference_errors, identity_errors = [], [], []
    for record in records.values():
        if "source_parity" in record:
            source_errors.append(max(record["source_parity"][name] for name in
                                     ("chosen_logit_abs_error", "logsumexp_abs_error",
                                      "logprob_abs_error", "top2_max_abs_error")))
        if "stage2_reference_errors" in record:
            reference_errors.append(max(record["stage2_reference_errors"].values()))
        if "identity_max_abs_error" in record:
            identity_errors.append(record["identity_max_abs_error"])
    _require((len(source_errors), len(reference_errors), len(identity_errors)) == (2, 4, 4) and
             max(source_errors + reference_errors + identity_errors) <= 2e-4 and
             all(not c["near_tie"] for r in results for c in r["matrix"]),
             "source/reference/identity or tie qualification changed")
    prior = m["budget"]["sequence_cumulative_prior_gpu_hours"]
    seconds = terminal["cost"]["allocated_gpu_seconds"]
    result = {"schema": "recurrence_chair_history_position.reduction.v1",
              "status": "candidate_not_lead_accepted", "protocol": literal_binding(PROTOCOL),
              "manifest": literal_binding(MANIFEST), "pilot": literal_binding(out / "pilot.json"),
              "terminal_receipt": literal_binding(out / "receipt.json"),
              "cold_readback": literal_binding(out / "cold-readback.json"),
              "cells_planned_executed_held": [18, 18, 0], "probes": results,
              "qualification": {"source_control_max_error": max(source_errors),
                                "stage2_reference_max_error": max(reference_errors),
                                "identity_full_vector_max_error": max(identity_errors),
                                "actual_consumer_crosschecks": checks,
                                "all_grid_gaps_over_0p001": True,
                                "numerical_contrast_guard": .0004},
              "cost": {"prior_sequence_gpu_hours": prior, "incremental_gpu_seconds": seconds,
                       "incremental_gpu_hours": seconds / 3600,
                       "cumulative_gpu_hours": prior + seconds / 3600,
                       "incremental_cap_gpu_hours": 1, "sequence_cap_gpu_hours": 8,
                       **terminal["cost"]},
              "terminal_job": {"pid": json.loads((out / "launch.json").read_text())["pid"],
                               "exit_code": 0, "status": terminal["status"],
                               "live_owned_gpu_jobs": [], "log": literal_binding(out / "gpu.log")},
              "artifact_bytes_before_reduction": sum(p.stat().st_size for p in out.rglob("*") if p.is_file())}
    _write_new(out / "reduction.json", result)
    return result


if __name__ == "__main__":
    result = reduce(OUTPUT)
    print(json.dumps({"status": result["status"], "cumulative_gpu_hours": result["cost"]["cumulative_gpu_hours"],
                      "prediction_checks": {x["candidate_owner"]: x["prediction_checks"] for x in result["probes"]}}))
