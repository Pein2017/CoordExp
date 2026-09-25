"""Fixed teacher-history address sensitivity on the retained 32 images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from src.artifacts.utf8_json import binding
from probes.coordinate_representation.coordinate_codebook_alignment import evaluation
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs, prepare_replay


def _ce_by_role(logits: torch.Tensor, targets: list[int], coord_ids: tuple[int, ...], wrapper_ids: set[int], eos_id: int) -> dict[str, object]:
    target = torch.tensor(targets, dtype=torch.long, device=logits.device)
    nll = F.cross_entropy(logits.float(), target, reduction="none").detach().cpu()
    coordinates = set(coord_ids)
    sums = {name: 0.0 for name in ("x1", "y1", "x2", "y2", "description")}
    counts = {name: 0 for name in sums}
    slot = 0
    for index, token in enumerate(targets):
        if token in coordinates:
            name = ("x1", "y1", "x2", "y2")[slot % 4]
            slot += 1
        elif token not in wrapper_ids and token != eos_id:
            name = "description"
        else:
            continue
        sums[name] += float(nll[index])
        counts[name] += 1
    if slot % 4:
        raise ValueError("frozen canonical response has an incomplete coordinate quartet")
    return {name: {"tokens": counts[name], "ce_sum": sums[name],
                   "ce_mean": sums[name] / counts[name] if counts[name] else None}
            for name in sums}


def run(plan_path: Path, admission_path: Path, output: Path) -> None:
    plan = json.loads(plan_path.read_text())
    if len(plan["cases"]) != 32 or plan["max_teacher_forced_forwards"] != 96:
        raise ValueError("address diagnostic plan is not the frozen 32 x 3 contract")
    checkpoint = Path(plan["checkpoint"])
    admission = json.loads(admission_path.read_text())
    by_dataset: dict[Path, dict[int, dict[str, object]]] = {}
    for case in plan["cases"]:
        if binding(Path(case["source_cell"]["path"])) != case["source_cell"]:
            raise ValueError("frozen diagnostic source-cell binding changed")
        dataset = Path(case["dataset"])
        if dataset not in by_dataset:
            by_dataset[dataset] = {int(row["image_id"]): row for row in evaluation._load_rows(dataset)}
    qwen, identity = evaluation._load_runtime(str(checkpoint), "cuda:0", admission)
    codebook = qwen.model.coordinate_codebook
    if codebook.mode != "early_patch_edges":
        raise ValueError("address diagnostic requires the trained early codebook")
    coord_ids = tuple(qwen.token_identity.coordinate_token_ids)
    from src.qwen.tokens import DEFAULT_WRAPPER_TOKENS
    wrapper_ids = {qwen.tokenizer.convert_tokens_to_ids(token) for token in DEFAULT_WRAPPER_TOKENS}
    wrapper_ids.update(qwen.token_identity.newline_token_ids)
    eos_id = qwen.tokenizer.convert_tokens_to_ids("<|im_end|>")
    rows = []
    forwards = 0
    for case in plan["cases"]:
        dataset = Path(case["dataset"])
        input_row = by_dataset[dataset][int(case["image_id"])]
        config = evaluation._config(admission, dataset)
        row = evaluation._normalize_case(qwen, input_row, config, dataset)
        if str(row["row_id"]) != case["row_id"]:
            raise ValueError("frozen diagnostic image identity changed")
        requests, _ = build_bound_native_requests(qwen, config, [row])
        batch = prepare_native_inputs(qwen.processor, requests,
                                      device=next(qwen.model.parameters()).device,
                                      record_media_identity=True)
        targets = evaluation._target_ids(qwen, row, config, dataset)
        replay = prepare_replay(qwen.model, batch.inputs,
                                prompt_token_ids=batch.prompt_token_ids[0],
                                continuation_token_ids=targets, compact_logits=True)
        grid = torch.tensor([batch.image_grids[0]], dtype=torch.long)
        edges = codebook.patch_edge_coordinates(grid, device=next(qwen.model.parameters()).device)
        count = int(edges.shape[0])
        shift = max(1, count // 2)
        codebook.capture_injection_stats = True
        with torch.inference_mode():
            correct = replay.aligned_logits(qwen.model(**replay.inputs).logits).float()
            correct_stats = dict(codebook.last_injection_stats or {})
            forwards += 1
            with codebook.override_patch_edges(edges):
                identity_logits = replay.aligned_logits(qwen.model(**replay.inputs).logits).float()
                identity_stats = dict(codebook.last_injection_stats or {})
            forwards += 1
            corrupted = None
            if count > 1:
                with codebook.override_patch_edges(edges.roll(shift, 0)):
                    corrupted = replay.aligned_logits(qwen.model(**replay.inputs).logits).float()
                    corrupt_stats = dict(codebook.last_injection_stats or {})
                forwards += 1
            else:
                corrupt_stats = None
        drift = float((correct - identity_logits).abs().max())
        if drift > 2e-4:
            raise AssertionError(f"identity replay drift exceeds 2e-4 for {case['row_id']}: {drift}")
        rows.append({
            "row_id": case["row_id"], "image_id": case["image_id"],
            "source_cell": case["source_cell"], "media_sha256": list(batch.media_sha256),
            "grid_thw": list(batch.image_grids[0]), "raw_patch_tokens": count,
            "shift": shift, "status": "complete" if corrupted is not None else "HOLD_singleton_grid",
            "identity_logit_max_abs": drift,
            "correct": {"ce": _ce_by_role(correct, targets, coord_ids, wrapper_ids, eos_id), "injection": correct_stats},
            "identity_replay": {"ce": _ce_by_role(identity_logits, targets, coord_ids, wrapper_ids, eos_id), "injection": identity_stats},
            "cyclic_shift": None if corrupted is None else {
                "ce": _ce_by_role(corrupted, targets, coord_ids, wrapper_ids, eos_id),
                "injection": corrupt_stats,
            },
        })
    codebook.capture_injection_stats = False
    result = {
        "schema": "early_edge.address_diagnostic.v1", "status": "complete",
        "scope": "same fixed teacher history; address corruption sensitivity only",
        "plan": binding(plan_path), "admission": binding(admission_path),
        "checkpoint_identity": identity, "teacher_forced_forwards": forwards,
        "max_teacher_forced_forwards": 96, "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.admission, args.output)
