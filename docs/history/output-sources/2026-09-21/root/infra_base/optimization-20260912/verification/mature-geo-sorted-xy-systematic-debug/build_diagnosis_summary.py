import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE_ROOT = Path(
    "/data/CoordExp/outputs/infra_base/optimization-20260912/verification/"
    "mature-geo-sorted-xy-consistency"
)
MECHANISM_ROOT = Path(
    "/data/CoordExp/outputs/infra_base/optimization-20260912/verification"
)

ARMS = {
    "A_dynamic_hf_bf16": {
        "role": "authoritative training-time computation graph",
        "backend": "hf",
        "precision": "bf16",
        "model_form": "base_plus_live_dora_plus_live_fp32_selected_token_delta",
        "metrics": BASELINE_ROOT / "lead-verification/evaluation/dynamic/metrics.json",
        "run": BASELINE_ROOT / "infer/dynamic-hf-bf16",
    },
    "B_dense_hf_bf16": {
        "role": "isolates dense BF16 materialization from the vLLM engine",
        "backend": "hf",
        "precision": "bf16",
        "model_form": "materialized_dense",
        "metrics": BASELINE_ROOT / "lead-verification/evaluation/materialized/metrics.json",
        "run": BASELINE_ROOT / "infer/materialized-hf-bf16",
    },
    "C_dense_vllm_bf16": {
        "role": "isolates the engine/path after materialization",
        "backend": "vllm",
        "precision": "bf16",
        "model_form": "same_materialized_dense_snapshot_as_B",
        "metrics": ROOT / "evaluation/vllm-bf16/metrics.json",
        "run": ROOT / "vllm-bf16",
    },
    "D_dense_hf_fp32": {
        "role": "higher-precision materialization intervention",
        "backend": "hf",
        "precision": "fp32",
        "model_form": "materialized_dense",
        "metrics": ROOT / "evaluation/hf-fp32/metrics.json",
        "run": ROOT / "hf-fp32",
    },
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def metrics(path: Path) -> dict:
    payload = load(path)
    return {
        "bbox_AP": payload["bbox_AP"],
        "bbox_AP50": payload["bbox_AP50"],
        "bbox_AP75": payload["bbox_AP75"],
        "row_count": payload["row_count"],
        "prediction_count": payload["coco_prediction_count"],
        "benchmark_eligible": payload["benchmark_eligible"],
    }


def delta_points(left: dict, right: dict) -> dict:
    return {
        key: (right[key] - left[key]) * 100
        for key in ("bbox_AP", "bbox_AP50", "bbox_AP75")
    }


def selected_image_plan(path: Path) -> list[dict]:
    fields = (
        "row_id",
        "image_path",
        "image_content_sha256",
        "executed_media_sha256",
        "declared_height",
        "declared_width",
        "decoded_height",
        "decoded_width",
        "expected_image_grid_thw",
        "logical_transform_id",
        "merged_visual_tokens",
        "backend_prompt_token_count",
    )
    return [
        {field: row.get(field) for field in fields}
        for row in map(json.loads, path.read_text().splitlines())
    ]


def main() -> None:
    arm_payload = {}
    for name, arm in ARMS.items():
        run_summary = load(arm["run"] / "summary.json")
        arm_payload[name] = {
            key: value for key, value in arm.items() if key not in {"metrics", "run"}
        }
        arm_payload[name]["artifact_dir"] = str(arm["run"])
        arm_payload[name]["metrics"] = metrics(arm["metrics"])
        arm_payload[name]["runtime_receipt"] = {
            "terminal_status": run_summary["terminal_status"],
            "decode_success_count": run_summary["decode_success_count"],
            "parser_failure_count": run_summary["parser_failure_count"],
            "score_failure_count": run_summary["score_failure_count"],
            "truncated_decode_count": run_summary["truncated_decode_count"],
        }

    a = arm_payload["A_dynamic_hf_bf16"]["metrics"]
    b = arm_payload["B_dense_hf_bf16"]["metrics"]
    c = arm_payload["C_dense_vllm_bf16"]["metrics"]
    d = arm_payload["D_dense_hf_fp32"]["metrics"]
    ab = delta_points(a, b)
    bc = delta_points(b, c)
    ac = delta_points(a, c)
    ad = delta_points(a, d)
    bd = delta_points(b, d)

    b_manifest = load(ARMS["B_dense_hf_bf16"]["run"] / "run_manifest.json")
    c_manifest = load(ARMS["C_dense_vllm_bf16"]["run"] / "run_manifest.json")
    exact_fields = (
        "dataset_identity",
        "generation_config_fingerprint",
        "generation_policy",
        "media_identity",
        "processor_identity_fingerprint",
        "prompt_policy_fingerprint",
        "prompt_trace",
        "score_policy_fingerprint",
        "template_identity",
        "tokenizer_identity",
    )
    b_model_path = b_manifest["backend_session"]["model_identity"]["base"]["path"]
    c_model_identity = c_manifest["backend_session"]["model_identity"]
    identity = {
        "B_and_C_same_dense_model_path": b_model_path == c_model_identity["model_path"],
        "B_and_C_composition_key": c_model_identity["composition_key"],
        "B_and_C_snapshot_fingerprint": c_model_identity["snapshot_fingerprint"],
        "B_and_C_exact_manifest_fields": {
            field: b_manifest[field] == c_manifest[field] for field in exact_fields
        },
        "B_and_C_selected_image_plan_fields_equal": selected_image_plan(
            ARMS["B_dense_hf_bf16"]["run"] / "image_plan.jsonl"
        )
        == selected_image_plan(
            ARMS["C_dense_vllm_bf16"]["run"] / "image_plan.jsonl"
        ),
        "boundary": (
            "The vLLM receipt does not expose an observed_image_grid_thw or a "
            "byte-level final projected visual tensor, so media/prompt identity "
            "does not prove visual-kernel tensor equality."
        ),
    }

    mechanism = load(
        MECHANISM_ROOT / "vllm-composition/diagnosis-summary.json"
    )
    fixed_prefix = load(
        MECHANISM_ROOT
        / "vllm-bounded-composition/final-prefix-diagnosis/diagnosis-summary.json"
    )
    paired_bc = load(
        ROOT / "analysis/materialized-hf-vs-vllm/paired_consistency.json"
    )["aggregate"]
    paired_ad = load(
        ROOT
        / "analysis/dynamic-hf-bf16-vs-materialized-hf-fp32/paired_consistency.json"
    )["aggregate"]
    excluded = load(ROOT / "metrics-excluding-shared-truncation.json")

    total_ap_gap = abs(ac["bbox_AP"])
    composition_ap_gap = abs(ab["bbox_AP"])
    payload = {
        "status": "diagnostic_completed_small_cohort",
        "repo_head": "86517a7f364b4f81b61d3765942f9a736262bd9e",
        "verdict": {
            "classification": (
                "partly_unavoidable_under_the_current_dense_bf16_graph_but_"
                "optimizable_not_fundamental"
            ),
            "exact_token_parity": (
                "not expected from the current dense BF16 composition plus "
                "vLLM path without preserving the dynamic arithmetic graph"
            ),
            "task_quality": (
                "the observed AP loss is optimizable; FP32 dense HF recovers "
                "most of the BF16 materialization loss on this cohort"
            ),
            "production_acceptance": "HOLD",
        },
        "scope": {
            "checkpoint": "four-coordinate-xy/step-2444",
            "cohort_rows": 32,
            "gt_objects": 296,
            "benchmark_eligible": False,
            "shared_truncated_row": "coco2017_val_000000076468",
            "claim_boundary": (
                "diagnostic sequential decomposition, not a full-validation "
                "quality bound or proof of additive causal effects"
            ),
        },
        "arms": arm_payload,
        "sequential_deltas_percentage_points": {
            "A_to_B_dense_bf16_materialization": ab,
            "B_to_C_vllm_engine_path": bc,
            "A_to_C_total_deployment_path": ac,
            "A_to_D_fp32_dense_intervention": ad,
            "B_to_D_higher_precision_dense": bd,
        },
        "descriptive_gap_accounting": {
            "composition_share_of_absolute_A_to_C_AP_gap": (
                composition_ap_gap / total_ap_gap
            ),
            "engine_path_share_of_absolute_A_to_C_AP_gap": (
                abs(bc["bbox_AP"]) / total_ap_gap
            ),
            "FP32_recovery_fraction_of_A_to_B_AP_gap": (
                bd["bbox_AP"] / abs(ab["bbox_AP"])
            ),
            "warning": (
                "shares are descriptive on one deterministic chain and should "
                "not be read as a universal causal attribution"
            ),
        },
        "robustness_excluding_shared_truncation": {
            "metrics": excluded,
            "A_to_B_AP_points": (
                excluded["arms"]["dense_hf_bf16"]["bbox_AP"]
                - excluded["arms"]["dynamic_hf_bf16"]["bbox_AP"]
            )
            * 100,
            "B_to_C_AP_points": (
                excluded["arms"]["dense_vllm_bf16"]["bbox_AP"]
                - excluded["arms"]["dense_hf_bf16"]["bbox_AP"]
            )
            * 100,
            "A_to_C_AP_points": (
                excluded["arms"]["dense_vllm_bf16"]["bbox_AP"]
                - excluded["arms"]["dynamic_hf_bf16"]["bbox_AP"]
            )
            * 100,
            "A_to_D_AP_points": (
                excluded["arms"]["dense_hf_fp32"]["bbox_AP"]
                - excluded["arms"]["dynamic_hf_bf16"]["bbox_AP"]
            )
            * 100,
        },
        "controlled_identity": identity,
        "paired_behavior": {
            "B_dense_hf_bf16_vs_C_dense_vllm_bf16": {
                "text_exact_rows": paired_bc["text_exact_equal_row_count"],
                "row_count": paired_bc["row_count"],
                "paired_predictions": paired_bc["paired_prediction_match_count"],
                "paired_iou_p50": paired_bc["paired_iou_distribution"]["p50"],
                "per_box_max_bin_delta_p50": paired_bc["bin_drift"][
                    "per_box_max_abs_delta"
                ]["distribution"]["p50"],
                "per_box_max_bin_delta_p95": paired_bc["bin_drift"][
                    "per_box_max_abs_delta"
                ]["distribution"]["p95"],
                "fraction_per_box_max_delta_at_most_5": paired_bc["bin_drift"][
                    "per_box_max_abs_delta"
                ]["fraction_at_most"]["at_most_5"]["fraction"],
                "unmatched_B": paired_bc["unmatched_dynamic_count"],
                "unmatched_C": paired_bc["unmatched_materialized_count"],
            },
            "A_dynamic_hf_bf16_vs_D_dense_hf_fp32": {
                "text_exact_rows": paired_ad["text_exact_equal_row_count"],
                "row_count": paired_ad["row_count"],
                "paired_predictions": paired_ad["paired_prediction_match_count"],
                "paired_iou_p50": paired_ad["paired_iou_distribution"]["p50"],
                "per_box_max_bin_delta_p50": paired_ad["bin_drift"][
                    "per_box_max_abs_delta"
                ]["distribution"]["p50"],
                "unmatched_A": paired_ad["unmatched_dynamic_count"],
                "unmatched_D": paired_ad["unmatched_materialized_count"],
            },
        },
        "mechanism_evidence": {
            "classification": mechanism["classification"],
            "actual_layer": mechanism["actual_layer_evidence"],
            "global_numeric": mechanism["global_numeric"],
            "secondary_graph_difference": mechanism[
                "secondary_known_graph_difference"
            ],
            "fixed_prefix_interpretation": fixed_prefix["interpretation"],
            "merge_device_counterfactual": (
                "CPU-merged and GPU-merged BF16 weights are each exactly equal "
                "to the published dense weight; changing merge device does not "
                "repair dynamic-vs-dense execution."
            ),
        },
        "FP32_intervention_caveat": (
            "D changes both weight/execution precision and HF attention "
            "implementation from FlashAttention2 to eager because FA2 does not "
            "support this FP32 diagnostic. It establishes optimizability but is "
            "not a single-factor proof that precision alone explains recovery."
        ),
        "evidence": {
            "source_session_rollout": (
                "/data/CoordExp/.codex/sessions/2026/09/12/"
                "rollout-2026-09-12T06-59-56-01a0946a-44a5-7b33-"
                "819e-fa172bb2c5c4_01a0946a-680d-75a1-a688-"
                "356a865cccf4.jsonl"
            ),
            "original_corrected_report": (
                "/data/CoordExp/.worktrees/coordexp-infras/research/"
                "investigations/ms-swift-upstream-comparison-2026-09-12/"
                "mature-geo-sorted-xy-consistency.md"
            ),
            "vllm_launch": str(ROOT / "vllm-bf16-diagnostic-launch.json"),
            "fp32_preparation": str(ROOT / "fp32-materialization-preparation.json"),
            "mechanism_diagnosis": str(
                MECHANISM_ROOT / "vllm-composition/diagnosis-summary.json"
            ),
            "fixed_prefix_diagnosis": str(
                MECHANISM_ROOT
                / "vllm-bounded-composition/final-prefix-diagnosis/"
                "diagnosis-summary.json"
            ),
        },
    }
    output = ROOT / "diagnosis-summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "verdict": payload["verdict"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
