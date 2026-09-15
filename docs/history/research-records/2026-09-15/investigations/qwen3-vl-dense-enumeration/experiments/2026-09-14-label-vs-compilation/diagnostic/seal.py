"""Seal the root-gated eight-cell free-h execution packet on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from probes.dora_owner_learning.candidate_opportunity import file_hash, require
from probes.parallel_owner_research.history import complete_rows
from probes.source_rweak_row_cross.run import native_record
from src.artifacts import publish_json_exclusive


REPO = Path("/data/CoordExp/.worktrees/research-probes")
BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
OLD = BASE / "2026-09-12-native-owner-scale-and-state/scale"
CURRENT = BASE / "2026-09-13-owner-successor-scale-throughput"
UNIT = REPO / (
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-09-14-label-vs-compilation"
)
OUTPUT = BASE / "2026-09-14-label-vs-compilation/diagnostic"
UNIT_MD = UNIT / "unit.md"
RUNNER = UNIT / "diagnostic/runner.py"
CANDIDATE = OUTPUT / "candidate-packet-v2.json"
INPUTS = OLD / "training/preparation/inputs-v2.json"
REVIEW = OLD / "visual-review-full-v2/final-reviews.json"
ACQUISITION = OLD / "acquisition-full-v2.json"
PAIRED = CURRENT / "evaluation/paired-preparation/packet-bound.json"
N16_ROWS = CURRENT / "evaluation/paired-preparation/n16-anchor-rows-896.jsonl"
A_ROWS = CURRENT / "evaluation/paired-consumer-v1/A-consumer.jsonl"
N16_COLD = OLD / "training/full-fixedP-N16-v2/cold-check.json"
A_COLD = CURRENT / "training/full-A-v2/cold-check.json"
TRAIN_RAW = BASE / "2026-09-05-sft256-dev128-baseline/inputs-v3/train.jsonl"
CAP = 3084
EOS = 151645


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing source: {path}")
    return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def without_eos(values: Sequence[int]) -> list[int]:
    ids = [int(value) for value in values]
    require(ids, "empty token sequence")
    if ids[-1] == EOS:
        ids.pop()
    require(EOS not in ids, "EOS inside token sequence")
    return ids


def literal_target(tokenizer: Any, ids: Sequence[int], frozen: Mapping[str, Any], label: str) -> dict[str, Any]:
    ids = [int(value) for value in ids]
    require(len(complete_rows(ids)) == 1, f"{label} is not one complete row")
    text = tokenizer.decode(ids, skip_special_tokens=False)
    require(tokenizer.encode(text, add_special_tokens=False) == ids, f"{label} tokenizer roundtrip")
    parsed = native_record(text, frozen["case"], frozen["golden"], "supplied_prefix")
    require(not parsed["dropped_predictions"] and len(parsed["pred"]) == 1, f"{label} native parser")
    pred = parsed["pred"][0]
    return {
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "description": pred["description"],
        "bbox_xyxy_pixels": list(pred["bbox"]),
        "coord_bins": list(pred["coord_bins"]),
    }


def prepare(output: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    candidate = read(CANDIDATE)
    require(candidate["schema"] == "label_vs_compilation.conditional_diagnostic_inventory.v1", "candidate schema")
    require(candidate["selection"]["selected_image_ids"] == [477415, 351017, 417044, 388795], "candidate cohort")
    require(candidate["selection"]["physical32_disjoint"], "candidate/physical32 disjointness")
    require(candidate["preparation_assertions"]["model_calls_performed"] == 0, "candidate preparation boundary")
    inputs = read(INPUTS)
    review = read(REVIEW)
    acquisition = read(ACQUISITION)
    paired = read(PAIRED)
    n16_rows = {int(row["image_id"]): row for row in read_jsonl(N16_ROWS)}
    a_rows = {int(row["image_id"]): row for row in read_jsonl(A_ROWS)}
    paired_rows = {int(row["image_id"]): row for row in paired["records"]}
    positive = {row["record_id"]: row for row in inputs["positive_records"]}
    conditional = {row["case_id"]: row for row in inputs["conditional_records"]}
    acquisition_rows = {row["job_id"]: row for row in acquisition["rows"]}
    n16_cold = read(N16_COLD)
    a_cold = read(A_COLD)
    require(n16_cold["status"] == a_cold["status"] == "passed", "endpoint cold scores")
    tokenizer = AutoTokenizer.from_pretrained(candidate["models"]["base_model_path"], local_files_only=True)

    cases = []
    cells = []
    for case_index, source in enumerate(candidate["cases"]):
        image_id = int(source["image_id"])
        entry = source["option_h_vs_h_plus_c"]
        c_record = positive[entry["c_record_id"]]
        w_record = conditional[c_record["case_id"]]
        frozen = paired_rows[image_id]
        require(frozen["evaluation_stratum"] == "admitted_target", "old-training stratum")
        require(c_record["example_id"] == frozen["example_id"] == source["example_id"], "example identity")
        require(c_record["prompt_token_ids"] == source["execution_identity"]["prompt_token_ids"], "prompt IDs")
        require(c_record["prompt_token_ids_sha256"] == n16_rows[image_id]["prompt_token_ids_sha256"] == a_rows[image_id]["prompt_token_ids_sha256"], "prompt digest")
        require(c_record["image"]["executed_media_sha256"] == n16_rows[image_id]["executed_media_sha256"] == a_rows[image_id]["executed_media_sha256"], "media digest")
        require(w_record["prefix_token_ids"] == c_record["prefix_token_ids"] + c_record["target_token_ids"], "exact h+c")
        require(review["decisions"][c_record["case_id"]]["status"] == "accept", "physical c+w admission")
        require(review["decisions"][c_record["case_id"]]["c_single_owner_absent_from_h"] is True, "physical c admission")
        require(review["decisions"][c_record["case_id"]]["w_single_owner_nonduplicate"] is True, "physical w admission")

        acquired = acquisition_rows[c_record["case_id"]]
        require(acquired["prefix_token_count"] == len(w_record["prefix_token_ids"]), "Stable50 h+c prefix count")
        require(acquired["prompt_token_ids_sha256"] == c_record["prompt_token_ids_sha256"], "Stable50 prompt identity")
        require(acquired["executed_media_sha256"] == c_record["image"]["executed_media_sha256"], "Stable50 media identity")
        free_rows = complete_rows(without_eos(acquired["free_token_ids"]))
        require(free_rows and free_rows[0] == w_record["target_token_ids"], "registered w is exact first Stable50 free row")
        require(acquired["local_w"]["status"] == "candidate_local_w", "registered immediate w")
        require(acquired["local_w"]["w_token_ids"] == w_record["target_token_ids"], "local w token identity")

        h_ids = [int(value) for value in c_record["prefix_token_ids"]]
        c_ids = [int(value) for value in c_record["target_token_ids"]]
        w_ids = [int(value) for value in w_record["target_token_ids"]]
        require(len(complete_rows(h_ids)) >= 1, "complete-row h")
        c_target = literal_target(tokenizer, c_ids, frozen, "c")
        w_target = literal_target(tokenizer, w_ids, frozen, "w")
        obligations = [
            {
                "owner_id": str(match["owner"]),
                "source_pred_index": int(match["pred_index"]),
                "source_iou": float(match["iou"]),
            }
            for match in acquired["free_score"]["50"]["matches"]
        ]
        require(len({item["owner_id"] for item in obligations}) == len(obligations), "owner obligation identity")

        case = {
            "case_index": case_index,
            "image_id": image_id,
            "example_id": c_record["example_id"],
            "source_case_id": c_record["case_id"],
            "prompt_token_ids": c_record["prompt_token_ids"],
            "prompt_token_ids_sha256": c_record["prompt_token_ids_sha256"],
            "executed_media_sha256": c_record["image"]["executed_media_sha256"],
            "observed_image_grid_thw": c_record["image"]["observed_image_grid_thw"],
            "image": c_record["image"],
            "h": {
                "token_ids": h_ids,
                "token_ids_sha256": digest(h_ids),
                "token_count": len(h_ids),
                "complete_rows": len(complete_rows(h_ids)),
                "credit": "supplied_no_free_credit",
            },
            "c": {
                "record_id": c_record["record_id"],
                "token_ids": c_ids,
                "token_ids_sha256": digest(c_ids),
                "token_count": len(c_ids),
                "literal_target": c_target,
                "physical_owner_provenance": c_record["candidate_owner_provenance"],
                "credit": "free_and_scored",
            },
            "w": {
                "record_id": w_record["record_id"],
                "token_ids": w_ids,
                "token_ids_sha256": digest(w_ids),
                "token_count": len(w_ids),
                "literal_target": w_target,
                "credit": "free_and_scored_if_generated",
            },
            "physical_review": review["decisions"][c_record["case_id"]],
            "annotation_owner_obligations": obligations,
            "obligation_source": {
                "checkpoint": "Stable50_only_not_N16_or_A_substitute",
                "acquisition_result": binding(ACQUISITION),
                "job_id": acquired["job_id"],
                "source_row_sha256": digest(acquired),
                "free_score_threshold": 0.5,
                "unknown_policy": "neutral",
            },
            "natural_root_context": {
                "role": "descriptive_existing_empty_root_not_matched_control",
                "N16": {
                    "source_row_sha256": digest(n16_rows[image_id]),
                    "tp50": n16_rows[image_id]["score"]["50"]["tp"],
                    "owners50": n16_rows[image_id]["score"]["50"]["owners"],
                    "parsed_prediction_count": n16_rows[image_id]["score"]["parsed_prediction_count"],
                    "parser_drops": n16_rows[image_id]["score"]["parser_drops"],
                    "stop_reason": n16_rows[image_id]["stop_reason"],
                },
                "A": {
                    "source_row_sha256": digest(a_rows[image_id]),
                    "tp50": a_rows[image_id]["score"]["50"]["tp"],
                    "owners50": a_rows[image_id]["score"]["50"]["owners"],
                    "parsed_prediction_count": a_rows[image_id]["score"]["parsed_prediction_count"],
                    "parser_drops": a_rows[image_id]["score"]["parser_drops"],
                    "stop_reason": a_rows[image_id]["stop_reason"],
                },
            },
        }
        cases.append(case)
        for arm in ("N16", "A"):
            scored = entry["c"][f"{arm}_final_literal_score"]
            cold = n16_cold if arm == "N16" else a_cold
            require(cold["positive_scores"][c_record["record_id"]]["argmax_target_tokens"] == len(c_ids), "cold c argmax")
            require(all(value == 0.0 for value in cold["live_score_deltas"][c_record["record_id"]].values()), "cold c score parity")
            cell_id = len(cells)
            cells.append({
                "cell_id": cell_id,
                "physical_gpu": cell_id,
                "arm": arm,
                "image_id": image_id,
                "example_id": case["example_id"],
                "adapter": candidate["models"][arm],
                "prompt_token_ids": case["prompt_token_ids"],
                "prompt_token_ids_sha256": case["prompt_token_ids_sha256"],
                "executed_media_sha256": case["executed_media_sha256"],
                "observed_image_grid_thw": case["observed_image_grid_thw"],
                "h": case["h"],
                "c": case["c"],
                "w": case["w"],
                "physical_review": case["physical_review"],
                "annotation_owner_obligations": obligations,
                "natural_root_context": case["natural_root_context"][arm],
                "scored_c": scored,
                "scientific_call": "one_complete_free_h_continuation",
                "supplied_credit": {"h": False, "c": True, "free_suffix": True},
                "prefix_token_count": len(h_ids),
                "remaining_budget": CAP - len(h_ids),
                "parity_stop": "If free IDs do not begin with exact c IDs, retain output and stop scientific interpretation for this cell; no automatic h+c call.",
            })

    require(len(cases) == 4 and len(cells) == 8, "frozen denominators")
    require([cell["physical_gpu"] for cell in cells] == list(range(8)), "GPU allocation")
    output_root = OUTPUT / "execution-v1"
    packet = {
        "schema": "label_vs_compilation.free_h_execution_packet.v1",
        "status": "sealed_root_launch_grant_pending",
        "unit_id": "2026-09-14-label-vs-compilation",
        "question": "On four exact old conditions h, does changing N16 to A preserve free complete-c realization and subsequent known-owner obligations despite natural-root coverage regression?",
        "sources": {
            "current_unit": binding(UNIT_MD),
            "archived_candidate_v2": binding(CANDIDATE),
            "sealer": binding(Path(__file__).resolve()),
            "runner": binding(RUNNER),
            "old_training_inputs": binding(INPUTS),
            "old_physical_review": binding(REVIEW),
            "Stable50_acquisition_obligation_source": binding(ACQUISITION),
            "Stable50_acquisition_packet": binding(Path(acquisition["packet"]["path"])),
            "paired_packet": binding(PAIRED),
            "N16_empty_root_rows": binding(N16_ROWS),
            "A_empty_root_rows": binding(A_ROWS),
            "N16_cold_scores": binding(N16_COLD),
            "A_cold_scores": binding(A_COLD),
            "materialization_train_raw": binding(TRAIN_RAW),
        },
        "models": {
            "N16": candidate["models"]["N16"],
            "A": candidate["models"]["A"],
            "base_model_path": candidate["models"]["base_model_path"],
            "source_embedding": candidate["models"]["source_embedding"],
        },
        "cases": cases,
        "cells": cells,
        "runtime": {
            "backend": "hf_native",
            "dtype": "fp32",
            "attention_implementation": "sdpa",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "eos_token_id": EOS,
            "total_assistant_token_cap_including_supplied_h": CAP,
            "strict_repeat": "any_class_iou_strictly_greater_than_0.95",
            "unknown_policy": "neutral",
            "forced_h_credit": False,
        },
        "bounds": {
            "complete_continuations": 8,
            "fallback_h_plus_c_calls": 0,
            "model_loads": 8,
            "image_forwards": 8,
            "max_new_tokens": sum(cell["remaining_budget"] for cell in cells),
            "max_model_forwards": sum(cell["remaining_budget"] + 1 for cell in cells),
            "max_rank_seconds": 24 * 60 * 60,
            "max_cuda_allocated_bytes_per_cell": 80 * 1024**3,
            "max_cuda_reserved_bytes_per_cell": 80 * 1024**3,
            "max_rss_bytes_per_cell": 96 * 1024**3,
            "retry_policy": "none_preserve_raw_and_terminal",
        },
        "tmux": {
            "session": "label-vs-compilation-diag-v1",
            "output_root": str(output_root),
            "launcher_log": str(OUTPUT / "execution-v1.launcher.log"),
            "launch_receipt": str(output_root / "process-launch.json"),
            "exit_receipt": str(output_root / "process-exits.json"),
            "terminal_receipt": str(output_root / "terminal.json"),
            "consumer_result": str(output_root / "result.json"),
            "expected_root_grant": str(OUTPUT / "root-launch-grant-v1.json"),
        },
        "consumer_contract": {
            "literal_c": "Exact free-token prefix equality to registered c; execution parity gate only.",
            "c_and_w_geometry": "Report best any-class and same-class IoU independently from literal equality; do not convert geometry into a new physical label.",
            "annotation_owners": "Report retained/missing/gained TP50 owners against prebound Stable50-source obligations; unknown remains neutral.",
            "burden": "Recompute native free/full score, parser burden, and strict any-class IoU>0.95 overlap ledger from raw IDs.",
            "cell_stop": "A free-c parity failure stops scientific interpretation only for that cell and never triggers an automatic h+c call.",
        },
        "interpretation_boundary": [
            "This is eight free-h continuations, not h-versus-h+c, history crossover, KV surgery, training, or a new natural-root rollout.",
            "Existing empty-root N16/A rows are descriptive reachability context, not an exchangeable matched control.",
            "Stable50 acquisition supplies frozen c/w and annotation-owner obligations only; its suffix is not an N16/A continuation substitute.",
            "The reviewed unlabeled donut c/w remains physical-review evidence with its original class/extent caveat, not a new GT annotation.",
            "Literal mismatch and geometric/owner fulfillment are reported separately; valid alternatives are not literal failures.",
            "No automatic fallback, case replacement, retry, training, crossover, or additional call follows an outcome.",
        ],
        "preparation": {
            "case_count": 4,
            "cell_count": 8,
            "model_calls": 0,
            "gpu_calls": 0,
            "root_launch_grant_present": False,
        },
    }
    require(packet["bounds"]["max_new_tokens"] == 24260, "exact token bound")
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_json_exclusive(output, packet)
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "execution-packet-v1.json")
    args = parser.parse_args()
    packet = prepare(args.output)
    print(json.dumps({
        "status": packet["status"],
        "output": str(args.output),
        "sha256": file_hash(args.output),
        "cases": len(packet["cases"]),
        "cells": len(packet["cells"]),
        "max_new_tokens": packet["bounds"]["max_new_tokens"],
        "model_calls": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
