"""Qualify the selected source reconstruction and row/branch slots without a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.artifacts.utf8_json import digest, literal_binding, write_pretty_json
from probes.model_profiles.mature_source import load_saved_source as _source
from probes.recurrence_dynamics.numerical_feedback.select import token_hash
from probes.recurrence_dynamics.recurrence_first_arrivals.prepare import BASE, CENSUS, MATURE, _require
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options


def row_trace_score(tokens: list[int], row: dict, trace: list[dict], batch_index: int) -> float:
    """Saved native full-vocabulary score with exact entry/terminator alignment."""
    start, end = int(row["start"]), int(row["end"])
    _require(tokens[start] == 151646 and tokens[end - 1] == 151649, "incomplete scored row")
    _require(tokens[start:end] == [int(step["chosen"][batch_index]) for step in trace[start:end]],
             "scored row token/trace alignment changed")
    _require(end - start == len(row["description_tokens"]) + 8, "row length excludes a grammar token")
    return sum(float(trace[index]["chosen_raw_logits"][batch_index])
               - float(trace[index]["logsumexp"][batch_index]) for index in range(start, end))


def extension(tokens: list[int], row: dict, candidate: dict, *, sham: bool) -> list[int]:
    j = candidate["first_differing_coordinate"]
    _require(j is not None and j < 3, "branch has no freely generated coordinate")
    position = int(row["coordinate_offsets"][j])
    expected = 151670 + int(candidate["bbox"][j])
    supplied = tokens[position] if sham else expected
    _require(tokens[position] != expected, "branch does not change the frozen coordinate")
    result = [*tokens[:position], supplied]
    _require(result[:position] == tokens[:position] and result[position] == supplied, "supplied slot mismatch")
    return result


def _row_from_saved(raw: list[int], proposal: dict) -> dict:
    from src.eval.numerical_recurrence import rows

    return rows(raw)[int(proposal["index"])]


def qualify(selection: Path) -> dict:
    proposal = json.loads(selection.read_text())
    _require(proposal["status"] == "proposed_lead_admission", "selection status changed")
    q = load_qwen_components_from_options(QwenLoadOptions(
        base_model=str(BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    _require(q.model is None and not q.load_model, "CPU qualification loaded a model")
    panels = {"mature": json.loads((MATURE / "panel.json").read_text()),
              "new": json.loads((CENSUS / "panel.json").read_text())}
    seen = {}
    family_checks = []
    branch_cells = {(cell["family"], cell["arrival_row"], cell["owner_id"]): cell
                    for cell in proposal["cells"] if cell["mode"] == "one_coordinate_supplied_branch"}
    for family in proposal["families"]:
        key = (family["source"], family["group"])
        source = family["source_bindings"]
        boundary = {"group": family["group"], "batch_index": family["batch_index"],
                    "image_id": family["image_id"], "raw_path": source["raw"]["path"],
                    "trace_path": source["trace"]["path"],
                    "receipt_path": source["runtime_receipt"]["path"],
                    "native_tokens": json.loads(Path(source["raw"]["path"]).read_text())["rows"][family["batch_index"]]["token_ids"]}
        boundary["native_token_hash"] = token_hash(boundary["native_tokens"])
        if key not in seen:
            batch, raw, trace, _group, planning = _source(boundary, "untied", panels[family["source"]], q, torch.device("cpu"))
            seen[key] = (batch, raw, trace, planning)
        batch, raw, trace, planning = seen[key]
        receipt = json.loads(Path(source["runtime_receipt"]["path"]).read_text())
        _require(digest(receipt["input_identity"]) == source["input_identity_sha256"], "reconstructed input identity differs")
        _require(batch.request_ids[family["batch_index"]] == raw[family["batch_index"]]["row_id"], "batch companion changed")
        native = raw[family["batch_index"]]["token_ids"]
        _require(native == boundary["native_tokens"], "native source tokens changed")
        step_trace = trace["steps"]
        scored = []
        for arrival in family["arrival_row_indices"]:
            proposed_row = family["row_evidence"][arrival]
            parsed = _row_from_saved(native, proposed_row)
            score = row_trace_score(native, parsed, step_trace, family["batch_index"])
            scored.append({"row_index": arrival, "native_trace_full_row_logp": score,
                           "first_token_offset": parsed["start"], "terminator_offset": parsed["end"] - 1})
            for candidate in family["candidates"]:
                planned = branch_cells.get((family["id"], arrival, candidate["owner_id"]))
                if planned is not None:
                    pair = {**candidate, "first_differing_coordinate": planned["coordinate_slot"]}
                    sham = extension(native, parsed, pair, sham=True)
                    branch = extension(native, parsed, pair, sham=False)
                    _require(sham[:-1] == branch[:-1] and sham[-1] != branch[-1], "branch changes more than one token")
                    _require(branch[-1] == planned["supplied_token_id"] and sham[-1] == planned["native_token_id"],
                             "cell supplied/native token differs from real branch caller")
        family_checks.append({"id": family["id"], "source_group": key, "replanned": planning["replanned_image_plan"],
                              "batch_size": len(raw), "image_grid": list(batch.image_grids[family["batch_index"]]),
                              "scores": scored})
    adapter = Path(proposal["families"][0]["source_bindings"]["adapter_path"])
    embedding = Path(proposal["families"][0]["source_bindings"]["embedding_delta_path"])
    payloads = [literal_binding(path) for path in (
        BASE / "config.json", BASE / "tokenizer.json",
        adapter / "adapter_config.json", adapter / "adapter_model.safetensors",
        embedding / "special_token_embeddings.json", embedding / "special_token_embeddings.safetensors")]
    source_bindings = [literal_binding(path) for path in (
        Path(__file__), Path(__file__).with_name("prepare.py"), Path(__file__).with_name("render_review.py"),
        Path("probes/model_profiles/mature_tied_untied.py"),
        Path('probes/recurrence_dynamics/coordinate_continuity/runtime.py'),
        Path('probes/recurrence_dynamics/numerical_feedback/select.py'),
        Path("src/qwen/generation.py"), Path("src/qwen/native.py"), Path("src/data/geometry.py"),
        Path("src/templates/renderer.py"))]
    return {"schema": "recurrence_first_arrivals.cpu_qualification.v1", "status": "passed",
            "selection": literal_binding(selection), "processor_only": True,
            "processor_identity": q.to_artifact_dict(), "payloads": payloads,
            "producer_import_bindings": source_bindings,
            "groups_reconstructed": len(seen), "families_checked": family_checks,
            "gpu_seconds": 0, "model_forwards": 0}


def selfcheck() -> None:
    tokens = [151646, 18921, 151647, 151648, 151670, 151771, 151872, 151973, 151649]
    row = {"start": 0, "end": 9, "index": 0, "description_tokens": [18921],
           "coordinate_offsets": [4, 5, 6, 7]}
    trace = [{"chosen": [token], "chosen_raw_logits": [0.0], "logsumexp": [1.0]} for token in tokens]
    _require(row_trace_score(tokens, row, trace, 0) == -9.0, "complete row score excludes entry/terminator")
    for bad_tokens, bad_row in ((tokens[:-1], {**row, "end": 8}), (tokens, {**row, "start": 1})):
        try:
            row_trace_score(bad_tokens, bad_row, trace, 0)
        except (IndexError, ValueError):
            pass
        else:
            raise AssertionError("dropped terminator or shifted slot was not rejected")
    candidate = {"bbox": [2, 101, 202, 303], "first_differing_coordinate": 0}
    sham = extension(tokens, row, candidate, sham=True)
    branch = extension(tokens, row, candidate, sham=False)
    _require(sham[-1] == tokens[4] and branch[-1] == 151672, "supply control failed")
    try:
        _require(sham == branch, "fake consumer ignored the supplied token")
    except ValueError:
        pass
    else:
        raise AssertionError("fake supplied-token consumer passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    selfcheck()
    if args.selfcheck:
        print("PASS row entry/terminator, shifted slot, and supplied-token checks")
        return
    if args.selection is None or args.output is None:
        parser.error("--selection and --output are required")
    _require(not args.output.exists(), "qualification output already exists")
    result = qualify(args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_pretty_json(args.output, result)
    print(json.dumps({"status": result["status"], "groups": result["groups_reconstructed"],
                      "families": len(result["families_checked"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
