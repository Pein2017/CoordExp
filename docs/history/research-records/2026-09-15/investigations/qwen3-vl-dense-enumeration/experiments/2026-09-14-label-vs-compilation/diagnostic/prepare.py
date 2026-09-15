"""Prepare the gated N16/A conditional-localization inventory without model work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from probes.parallel_owner_research.history import complete_rows
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
SYNTHESIS = (
    REPO
    / "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-09-13-owner-successor-scale-throughput/consultation-2026-09-14/synthesis.md"
)
RETAINED = SYNTHESIS.parent / "root-retained-evidence.json"
SUPPLY = UNIT / "supply/result.json"
INPUTS = OLD / "training/preparation/inputs-v2.json"
N16_RECEIPT = OLD / "training/full-fixedP-N16-v2/receipt.json"
A_RECEIPT = CURRENT / "training/full-A-v2/receipt.json"
PAIRED_PACKET = CURRENT / "evaluation/paired-preparation/packet-bound.json"
N16_ROWS = CURRENT / "evaluation/paired-preparation/n16-anchor-rows-896.jsonl"
A_ROWS = CURRENT / "evaluation/paired-consumer-v1/A-consumer.jsonl"
OLD_REVIEW = OLD / "visual-review-full-v2/final-reviews.json"
BLIND32 = (
    CURRENT
    / "evaluation/paired-consumer-v1/blind32-review-preparation-v1/manifest.json"
)
PRIOR_HISTORY_PACKET = CURRENT / "history/preparation/packet.json"
PRIOR_HISTORY_RESULT = CURRENT / "history/launch/result.json"

CAP = 3084
EOS = 151645

# Deterministic preferred c records for the entry diagnostic. Each is required
# to be a complete old physical c+w package and all-token argmax at both N16/A.
ENTRY_RECORDS = {
    351017: "coco2017_train_000000351017:499060:c",
    388795: "coco2017_train_000000388795:1986212:c",
    417044: "coco2017_train_000000417044:417044:review:P6:c",
    477415: "coco2017_train_000000477415:1309807:c",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing source: {path}")
    return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def without_eos(values: Sequence[int]) -> list[int]:
    ids = [int(value) for value in values]
    require(ids, "empty action IDs")
    if ids[-1] == EOS:
        ids.pop()
    require(EOS not in ids, "EOS inside action IDs")
    return ids


def source_rows(row: Mapping[str, Any]) -> list[list[int]]:
    return complete_rows(without_eos(row["action_ids"]))


def row_index(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {int(row["image_id"]): row for row in rows}
    require(len(result) == len(rows), f"duplicate image in {path}")
    return result


def score_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "token_count": int(value["token_count"]),
        "argmax_target_tokens": int(value["argmax_target_tokens"]),
        "mean_logprob": float(value["mean_logprob"]),
        "min_target_margin": float(value["min_target_margin"]),
    }


def owners_after(row: Mapping[str, Any], boundary: int) -> list[dict[str, Any]]:
    return [
        {"owner_id": str(match["owner"]), "pred_index": int(match["pred_index"])}
        for match in row["score"]["50"]["matches"]
        if int(match["pred_index"]) > boundary
    ]


def history_descriptor(
    *, row: Mapping[str, Any], rows: Sequence[Sequence[int]], boundary: int, source_arm: str
) -> dict[str, Any]:
    prefix = [token for item in rows[: boundary + 1] for token in item]
    full = [int(value) for value in row["action_ids"]]
    require(full[: len(prefix)] == prefix, f"{source_arm} prefix is not literal action prefix")
    suffix = full[len(prefix) :]
    require(suffix, f"{source_arm} history is terminal")
    return {
        "source_arm": source_arm,
        "boundary": "after_first_differing_complete_row_before_selected_incumbent",
        "complete_rows": boundary + 1,
        "prefix_token_ids": prefix,
        "prefix_token_ids_sha256": digest(prefix),
        "prefix_token_count": len(prefix),
        "remaining_budget": CAP - len(prefix),
        "archived_diagonal_free_token_ids": suffix,
        "archived_diagonal_free_token_ids_sha256": digest(suffix),
        "archived_diagonal_free_token_count": len(suffix),
        "archived_stop_reason": row["stop_reason"],
        "source_action_ids_sha256": digest(full),
        "source_row_sha256": digest(row),
    }


def prepare(output: Path) -> dict[str, Any]:
    inputs = read(INPUTS)
    n16_receipt = read(N16_RECEIPT)
    a_receipt = read(A_RECEIPT)
    paired = read(PAIRED_PACKET)
    review = read(OLD_REVIEW)
    retained = read(RETAINED)
    supply = read(SUPPLY)
    blind32 = read(BLIND32)
    n16_by_id = row_index(N16_ROWS)
    a_by_id = row_index(A_ROWS)
    packet_by_id = {int(row["image_id"]): row for row in paired["records"]}
    require(len(packet_by_id) == len(paired["records"]), "paired packet image identity")

    positive_by_id = {row["record_id"]: row for row in inputs["positive_records"]}
    conditional_by_case = {row["case_id"]: row for row in inputs["conditional_records"]}
    require(len(positive_by_id) == len(inputs["positive_records"]), "positive record identity")
    require(len(conditional_by_case) == len(inputs["conditional_records"]), "conditional case identity")

    train11 = sorted(int(image_id) for image_id in paired["strata"]["train11"]["image_ids"])
    require(len(train11) == 11, "expected exact train11 stratum")
    candidates: list[tuple[int, int]] = []
    census: list[dict[str, Any]] = []
    for image_id in train11:
        n16 = n16_by_id[image_id]
        arm_a = a_by_id[image_id]
        delta = int(n16["score"]["50"]["tp"]) - int(arm_a["score"]["50"]["tp"])
        reasons = []
        if delta <= 0:
            reasons.append("no_positive_N16_minus_A_TP50_regression")
        if int(n16["score"]["parser_drops"]) or int(arm_a["score"]["parser_drops"]):
            reasons.append("parser_drop_in_archived_root_output")
        if not reasons:
            # This validates complete literal storage rather than repairing a tail.
            require(len(source_rows(n16)) == int(n16["score"]["parsed_prediction_count"]), "N16 row count")
            require(len(source_rows(arm_a)) == int(arm_a["score"]["parsed_prediction_count"]), "A row count")
            candidates.append((image_id, delta))
        census.append(
            {
                "image_id": image_id,
                "N16_tp50": int(n16["score"]["50"]["tp"]),
                "A_tp50": int(arm_a["score"]["50"]["tp"]),
                "N16_minus_A_tp50": delta,
                "N16_parser_drops": int(n16["score"]["parser_drops"]),
                "A_parser_drops": int(arm_a["score"]["parser_drops"]),
                "disposition": "eligible" if not reasons else "excluded",
                "reasons": reasons,
            }
        )
    candidates.sort(key=lambda item: (-item[1], item[0]))
    selected_ids = [image_id for image_id, _ in candidates[:4]]
    require(selected_ids == [477415, 351017, 417044, 388795], "frozen selection changed")

    blind_ids = {int(image_id) for batch in blind32["batches"] for image_id in batch["images"]}
    require(len(blind_ids) == int(blind32["images"]) == 32, "physical32 identity")
    require(set(selected_ids).isdisjoint(blind_ids), "old training candidate overlaps physical32")

    cases = []
    history_max = 0
    entry_max = 0
    for image_id in selected_ids:
        n16 = n16_by_id[image_id]
        arm_a = a_by_id[image_id]
        record = packet_by_id[image_id]
        require(record["evaluation_stratum"] == "admitted_target", "candidate left train11/admitted-target stratum")
        require(n16["prompt_token_ids_sha256"] == arm_a["prompt_token_ids_sha256"], "prompt mismatch")
        require(n16["executed_media_sha256"] == arm_a["executed_media_sha256"], "media mismatch")
        require(n16["adapter_fingerprint"] == paired["anchor"]["adapter_fingerprint"], "N16 adapter mismatch")
        require(arm_a["adapter_fingerprint"] == paired["endpoints"]["A"]["adapter"]["fingerprint"], "A adapter mismatch")

        n16_rows = source_rows(n16)
        a_rows = source_rows(arm_a)
        first_diff = next(
            (index for index, pair in enumerate(zip(n16_rows, a_rows)) if pair[0] != pair[1]),
            min(len(n16_rows), len(a_rows)),
        )
        require(first_diff < min(len(n16_rows), len(a_rows)), "no common-position row divergence")
        require(first_diff + 1 < len(n16_rows) and first_diff + 1 < len(a_rows), "terminal crossover boundary")

        a_owners = {str(owner) for owner in arm_a["score"]["50"]["owners"]}
        lost = [item for item in owners_after(n16, first_diff) if item["owner_id"] not in a_owners]
        lost.sort(key=lambda item: (item["pred_index"], item["owner_id"]))
        require(lost, "no N16 suffix incumbent absent from A")
        selected_target = lost[0]
        require(selected_target["pred_index"] > first_diff, "target supplied in history")
        suffix_sources: dict[str, list[dict[str, Any]]] = {
            "N16": owners_after(n16, first_diff),
            "A": owners_after(arm_a, first_diff),
        }
        all_obligations: dict[str, set[str]] = {}
        for source_arm, values in suffix_sources.items():
            for item in values:
                all_obligations.setdefault(item["owner_id"], set()).add(source_arm)

        histories = {
            "N16_actual": history_descriptor(
                row=n16, rows=n16_rows, boundary=first_diff, source_arm="N16"
            ),
            "A_actual": history_descriptor(
                row=arm_a, rows=a_rows, boundary=first_diff, source_arm="A"
            ),
        }
        history_case_max = 2 * sum(item["remaining_budget"] for item in histories.values())
        history_max += history_case_max

        positive = positive_by_id[ENTRY_RECORDS[image_id]]
        conditional = conditional_by_case[positive["case_id"]]
        require(conditional["prefix_token_ids"] == positive["prefix_token_ids"] + positive["target_token_ids"], "h+c identity")
        require(conditional["prompt_token_ids_sha256"] == positive["prompt_token_ids_sha256"], "entry prompt mismatch")
        require(positive["prompt_token_ids_sha256"] == n16["prompt_token_ids_sha256"], "entry/root prompt mismatch")
        require(positive["image"]["executed_media_sha256"] == n16["executed_media_sha256"], "entry/root media mismatch")
        require(review["decisions"][positive["case_id"]]["status"] == "accept", "entry physical review")
        require(review["decisions"][positive["case_id"]]["c_single_owner_absent_from_h"] is True, "c admission")
        require(review["decisions"][positive["case_id"]]["w_single_owner_nonduplicate"] is True, "w admission")
        score_n16 = n16_receipt["final_live_positive_scores"][positive["record_id"]]
        score_a = a_receipt["final_live_positive_scores"][positive["record_id"]]
        require(int(score_n16["argmax_target_tokens"]) == len(positive["target_token_ids"]), "N16 literal c not all argmax")
        require(int(score_a["argmax_target_tokens"]) == len(positive["target_token_ids"]), "A literal c not all argmax")
        h_ids = [int(value) for value in positive["prefix_token_ids"]]
        c_ids = [int(value) for value in positive["target_token_ids"]]
        w_ids = [int(value) for value in conditional["target_token_ids"]]
        require(len(complete_rows(h_ids)) >= 1, "entry h is not complete-row history")
        require(len(complete_rows(c_ids)) == 1 and len(complete_rows(w_ids)) == 1, "entry c/w literal shape")
        require(len(h_ids) + len(c_ids) < CAP, "entry h+c exhausts budget")
        entry_case_max = 2 * ((CAP - len(h_ids)) + (CAP - len(h_ids) - len(c_ids)))
        entry_max += entry_case_max

        cases.append(
            {
                "priority": len(cases) + 1,
                "case_id": f"train11:{image_id}",
                "image_id": image_id,
                "example_id": n16["example_id"],
                "selection_evidence": {
                    "stratum": "train11_old_training_not_confirmation32",
                    "N16_tp50": int(n16["score"]["50"]["tp"]),
                    "A_tp50": int(arm_a["score"]["50"]["tp"]),
                    "N16_minus_A_tp50": int(n16["score"]["50"]["tp"]) - int(arm_a["score"]["50"]["tp"]),
                    "N16_complete_rows": len(n16_rows),
                    "A_complete_rows": len(a_rows),
                    "parser_drops": {"N16": 0, "A": 0},
                    "stop_reasons": {"N16": n16["stop_reason"], "A": arm_a["stop_reason"]},
                    "N16_source_row_sha256": digest(n16),
                    "A_source_row_sha256": digest(arm_a),
                },
                "execution_identity": {
                    "prompt_token_ids": record["prompt_token_ids"],
                    "prompt_token_ids_sha256": n16["prompt_token_ids_sha256"],
                    "executed_media_sha256": n16["executed_media_sha256"],
                    "image": positive["image"],
                },
                "option_history_crossover": {
                    "eligibility": "root_ruling_required_that_archived_N16_suffix_is_whole_useful_and_A_root_is_bad",
                    "first_differing_complete_row_index_0_based": first_diff,
                    "histories": histories,
                    "selected_known_N16_suffix_owner_absent_from_A": selected_target,
                    "all_annotation_relative_free_suffix_owner_obligations": [
                        {"owner_id": owner, "archived_source_arms": sorted(source_arms)}
                        for owner, source_arms in sorted(all_obligations.items())
                    ],
                    "archived_free_suffix_owner_matches_by_history": suffix_sources,
                    "calls_if_selected": 4,
                    "call_cells": [
                        {"recipient_model": model, "history": history}
                        for model in ("N16", "A")
                        for history in ("N16_actual", "A_actual")
                    ],
                    "max_new_tokens_after_prefix_deductions": history_case_max,
                    "diagonal_requirement": "Fresh-prefill N16/N16-history and A/A-history must reproduce each archived literal free suffix and stop before off-diagonal interpretation.",
                    "credit_rule": "No supplied history row earns free credit; alternatives are accepted through native parser plus annotation/physical-owner matching, not literal-only equality.",
                },
                "option_h_vs_h_plus_c": {
                    "eligibility": "admissible_when_whole_conditional_usefulness_is_unestablished",
                    "retained_evidence_limit": "Physical c and immediate w are accepted and c is literal all-token argmax at N16/A; neither fact establishes a useful whole free continuation.",
                    "source_case_id": positive["case_id"],
                    "c_record_id": positive["record_id"],
                    "w_record_id": conditional["record_id"],
                    "candidate_owner_provenance": positive["candidate_owner_provenance"],
                    "physical_review": review["decisions"][positive["case_id"]],
                    "h": {
                        "token_ids": h_ids,
                        "token_ids_sha256": digest(h_ids),
                        "token_count": len(h_ids),
                        "complete_rows": len(complete_rows(h_ids)),
                    },
                    "c": {
                        "token_ids": c_ids,
                        "token_ids_sha256": digest(c_ids),
                        "token_count": len(c_ids),
                        "owner_id": str(positive["candidate_owner_provenance"]["owner_id"]),
                        "N16_final_literal_score": score_projection(score_n16),
                        "A_final_literal_score": score_projection(score_a),
                    },
                    "w": {
                        "token_ids": w_ids,
                        "token_ids_sha256": digest(w_ids),
                        "token_count": len(w_ids),
                        "unknown_mask_policy": conditional["unknown_mask_policy"],
                    },
                    "calls_if_selected": 4,
                    "call_cells": [
                        {
                            "recipient_model": model,
                            "prefix": prefix,
                            "prefix_token_count": len(h_ids) + (len(c_ids) if prefix == "h_plus_c" else 0),
                            "remaining_budget": CAP - len(h_ids) - (len(c_ids) if prefix == "h_plus_c" else 0),
                        }
                        for model in ("N16", "A")
                        for prefix in ("h_free", "h_plus_c")
                    ],
                    "max_new_tokens_after_prefix_deductions": entry_case_max,
                    "credit_rule": "Under h, c and later free owners may earn credit. Under h+c, c is supplied and earns no credit; only the free suffix is scored. Alternative valid rows/geometry remain eligible.",
                },
            }
        )

    prior_history = read(PRIOR_HISTORY_RESULT)
    prior_cost = prior_history["cost"]
    require(int(retained["records"]) == 69 and int(retained["unique_image_prompt_prefix_conditions"]) == 56, "retained score denominator")
    require(int(supply["acquisition"]["frozen_images"]) == 4096, "supply frozen denominator")
    require(int(supply["physical_admission"]["final_packages"]) == 53, "supply package denominator")
    require(int(supply["physical_admission"]["final_images"]) == 39, "supply image denominator")

    packet = {
        "schema": "label_vs_compilation.conditional_diagnostic_inventory.v1",
        "status": "candidate_cpu_verified_root_surface_grant_required",
        "unit_id": "2026-09-14-label-vs-compilation",
        "question": "For a small clean old-training regression panel, does failure localize to free entry/conditional continuation or to recipient compatibility with an exact actual history?",
        "sources": {
            "unit": binding(UNIT_MD),
            "consultation_synthesis": binding(SYNTHESIS),
            "retained_literal_scores": binding(RETAINED),
            "supply_audit": binding(SUPPLY),
            "old_training_inputs": binding(INPUTS),
            "old_physical_review": binding(OLD_REVIEW),
            "N16_training_receipt": binding(N16_RECEIPT),
            "A_training_receipt": binding(A_RECEIPT),
            "paired_packet": binding(PAIRED_PACKET),
            "N16_root_rows": binding(N16_ROWS),
            "A_root_rows": binding(A_ROWS),
            "physical32_manifest": binding(BLIND32),
            "prior_analog_history_packet": binding(PRIOR_HISTORY_PACKET),
            "prior_analog_history_result": binding(PRIOR_HISTORY_RESULT),
            "producer": binding(Path(__file__)),
        },
        "models": {
            "N16": paired["model"]["current_adapter"],
            "A": paired["endpoints"]["A"]["adapter"],
            "base_model_path": paired["model"]["base_model_path"],
            "source_embedding": paired["model"]["source_embedding"],
        },
        "selection": {
            "rule": "Within exact train11, require N16 TP50>A TP50, zero parser drops, literal complete archived root rows, nonterminal first-divergence crossover, and an old physically accepted c+w record whose c is all-token argmax at both endpoints. Rank by descending N16-minus-A TP50 then image ID; take at most four; no backfill.",
            "train11_census": census,
            "selected_image_ids": selected_ids,
            "selected_image_ids_sha256": digest(selected_ids),
            "simplest_single_case_candidate": {
                "image_id": selected_ids[0],
                "reason": "largest clean N16-minus-A TP50 regression under the frozen rule",
                "calls_after_root_selects_one_surface": 4,
            },
            "physical32_disjoint": True,
            "physical32_image_ids_sha256": digest(sorted(blind_ids)),
            "physical32_role": "confirmation-only; no label or physical-owner evidence is borrowed for these old train11 cases",
        },
        "surface_gate": {
            "selected_surface": None,
            "root_must_select_exactly_one": ["h_vs_h_plus_c", "actual_history_crossover"],
            "current_decision_evidence": {
                "h_vs_h_plus_c": "Admissible now if conditional usefulness remains unknown; retained evidence is only physical c+immediate-w admission and literal c scoring.",
                "actual_history_crossover": "Conditionally eligible only if root accepts an archived N16 free suffix as whole useful and the A root outcome as bad; GT50 retained owner matches alone are not a new physical review.",
            },
            "not_authorized": "No model call exists until root binds one surface, case count, exact execution packet, and named detached tmux artifacts.",
        },
        "cases": cases,
        "runtime_contract": {
            "models": ["N16", "A"],
            "backend": "hf_native",
            "dtype": "fp32",
            "attention_implementation": "sdpa",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "eos_token_id": EOS,
            "total_assistant_token_cap_including_supplied_history": CAP,
            "parser_geometry_matcher": "existing native parser and original geometry; strict any-class repeat iff IoU>0.95; unknown is neutral",
            "forced_credit": False,
            "complete_continuation_calls_max": 16,
            "unprefixed_envelope_before_prefix_deductions": 16 * CAP,
            "exact_max_new_tokens_after_prefix_deductions_if_all_four_h_vs_h_plus_c": entry_max,
            "exact_max_new_tokens_after_prefix_deductions_if_all_four_history_crossover": history_max,
            "no_outcome_backfill": True,
        },
        "retained_evidence": {
            "literal_score_conditions": {
                "records": retained["records"],
                "unique_conditions": retained["unique_image_prompt_prefix_conditions"],
                "multiple_target_conditions": retained["multiple_distinct_literal_target_conditions"],
                "A_old_conditions_with_any_complete_literal_all_argmax": retained["arms"]["A"]["banks"]["old"]["conditions_with_any_literal_full_argmax"],
            },
            "supply": {
                "frozen_images": supply["acquisition"]["frozen_images"],
                "images_with_gt_backed_nomination": supply["acquisition"]["images_with_gt_backed_nomination"],
                "images_with_candidate_local_w": supply["conditional_immediate_witness"]["images_with_any_candidate_local_w"],
                "final_packages": supply["physical_admission"]["final_packages"],
                "final_images": supply["physical_admission"]["final_images"],
                "selected_packages_with_machine_burden": supply["later_continuation_machine_burden"]["selected_packages"]["any_machine_burden"],
                "unknown_neutral_groups": supply["physical_admission"]["root_disposition_counts"]["hold"],
                "limit": "No identified missing-label counterfactual or missing-owner list; immediate w admission is not whole-suffix usefulness.",
            },
            "endpoint_states": "Only final A256/B256 and an independent smoke adapter exist; no A32/A128 optimizer or adapter state is available and no interpolation is allowed.",
        },
        "cost_bound": {
            "calls": {"full_inventory_after_one_surface": 16, "simplest_single_case": 4},
            "hard_new_token_envelope_before_prefix_deductions": 16 * CAP,
            "prior_analog_16_call_observation": {
                "source": binding(PRIOR_HISTORY_RESULT),
                "gpu_hours": prior_cost["total_gpu_hours"],
                "model_forwards": prior_cost["total_model_forwards"],
                "image_forwards": prior_cost["total_image_forwards"],
                "peak_cuda_allocated_bytes": max(
                    arm["peak_cuda_allocated_bytes"] for arm in prior_cost["arms"].values()
                ),
                "peak_cuda_reserved_bytes": max(
                    arm["peak_cuda_reserved_bytes"] for arm in prior_cost["arms"].values()
                ),
                "qualification": "planning evidence only; different endpoints/cases may cost differently and no utilization is inferred",
            },
        },
        "future_execution": {
            "runner_requirement": "After root surface selection, add the smallest unit-local N16/A consumer around existing native generate_continuations/fresh-prefill patterns; existing history.py is hard-coded to Stable50/N16 and must not be invoked unchanged.",
            "tmux_session": "label-vs-compilation-diag-v1",
            "output_root": str(OUTPUT / "execution-v1"),
            "launcher_log": str(OUTPUT / "execution-v1/launcher.log"),
            "terminal_receipt": str(OUTPUT / "execution-v1/terminal.json"),
            "state": "paths_reserved_only_no_command_no_process_no_launch",
        },
        "risks_and_stop": [
            "Diagonal fresh-prefill literal suffix mismatch invalidates history off-diagonal interpretation and stops that surface.",
            "A valid alternative row/string/geometry can satisfy owner obligations; literal mismatch alone is not failure.",
            "An invalid, malformed, incomplete, or EOS-only continuation is retained as an inactive/failed case, never repaired or replaced.",
            "The strict repeat threshold is IoU>0.95 any class; IoU==0.95 is not a repeat and unknown remains neutral.",
            "Stop after the one root-selected surface, at most the frozen case count and 16 complete calls; do not add the other surface or outcome-driven cases.",
        ],
        "root_rulings_remaining": [
            "Choose exactly one surface: h versus h+c while whole conditional usefulness is unknown, or actual-history crossover only after accepting useful-old-suffix/bad-A eligibility.",
            "Choose one to four leading inventory cases before execution; the simplest candidate is image 477415 (four calls).",
            "Bind a unit-local runner packet and named tmux command/log/terminal receipt; this CPU inventory is not a launch grant.",
        ],
        "preparation_assertions": {
            "case_count": len(cases),
            "max_case_count": 4,
            "calls_per_case_after_one_surface": 4,
            "max_complete_calls": 16,
            "model_calls_performed": 0,
            "gpu_calls_performed": 0,
            "new_labels_created": 0,
            "training_updates": 0,
        },
    }
    require(len(packet["cases"]) <= 4, "case cap")
    require(len(packet["cases"]) * 4 <= 16, "continuation cap")
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_json_exclusive(output, packet)
    return packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT / "candidate-packet-v1.json")
    args = parser.parse_args()
    packet = prepare(args.output)
    print(
        json.dumps(
            {
                "status": packet["status"],
                "output": str(args.output),
                "sha256": file_hash(args.output),
                "selected_image_ids": packet["selection"]["selected_image_ids"],
                "case_count": len(packet["cases"]),
                "max_calls_after_one_surface": len(packet["cases"]) * 4,
                "model_calls_performed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
