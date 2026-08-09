from __future__ import annotations

import copy
from importlib import metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.research import seal_s_natural_boundary_k_n_h_pre_gpu_receipt as sealer
from scripts.research import materialize_s_natural_boundary_k_n_h_pre_gpu_evidence as materializer


def _write_json(path: Path, document: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(sealer.canonical_json_bytes(document) + b"\n")
    return path


def _self(document: dict[str, Any], field: str = "self_sha256") -> dict[str, Any]:
    result = dict(document)
    body = dict(result)
    body.pop(field, None)
    result[field] = sealer.sha256_json(body)
    return result


def _file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = tmp_path / "fixture"
    root.mkdir(parents=True)
    authority_path = _file(root / "authority.md", "active S cohort authority\n")
    authority = {
        "path": str(authority_path),
        "sha256": sealer.sha256_file(authority_path),
        "unit_id": sealer.UNIT_ID,
        "status": "active",
        "scope": "one full admitted S K/N/H cohort",
    }

    gate_runtime = {
        "schema_version": "s_primary_natural_boundary_gate.v1.runtime_identity.v1",
        "unit_id": sealer.UNIT_ID,
        "checkpoint": "S",
        "event_id": "gt:5001:15",
        "gpu_launch_authorized": False,
        "no_training": True,
        "event": {
            "checkpoint": "S",
            "event_id": "gt:5001:15",
        },
    }
    gate_runtime["identity_sha256"] = sealer.sha256_json(gate_runtime)
    gate_runtime_path = _write_json(root / "gate" / "runtime_identity.json", gate_runtime)
    gate_arm_counts = {
        arm: (27 if index < 11 else 1)
        for index, arm in enumerate(sealer.ARM_ORDER)
    }
    gate_result = {
        "schema_version": "s_primary_natural_boundary_gate.v1",
        "unit_id": sealer.UNIT_ID,
        "checkpoint": "S",
        "event_id": "gt:5001:15",
        "gpu_launch_authorized": False,
        "no_training": True,
        "arm_order": list(sealer.ARM_ORDER),
        "arms": {
            arm: {
                "scalar_forward_count": count,
                "runtime_scalar_forward_count": count,
            }
            for arm, count in gate_arm_counts.items()
        },
        "runtime_identity_sha256": gate_runtime["identity_sha256"],
    }
    gate_result["result_sha256"] = sealer.sha256_json(gate_result)
    gate_result_path = _write_json(root / "gate" / "result.json", gate_result)
    gate_terminal_path = _write_json(
        root / "gate" / "terminal_summary.json",
        {
            "schema_version": "s_primary_natural_boundary_gate.v1.terminal.v1",
            "status": "completed",
            "checkpoint": "S",
            "event_id": "gt:5001:15",
            "result_sha256": gate_result["result_sha256"],
        },
    )
    gate_log = _file(root / "gate" / "launch.log", "completed\n")
    gate_artifacts = {
        "result": gate_result_path,
        "runtime_identity": gate_runtime_path,
        "terminal_summary": gate_terminal_path,
        "launch_log": gate_log,
    }

    support_plan = {
        "schema_version": "natural_boundary_owner_support_completion_plan.v1",
        "status": "sealed_cpu_plan",
        "unit_id": sealer.UNIT_ID,
        "checkpoint": "S",
        "contexts": [],
    }
    support_plan["plan_content_sha256"] = sealer.sha256_json(support_plan)
    support_plan_path = _write_json(root / "support" / "plan.json", support_plan)
    census_v2_path = _write_json(
        root / "support" / "census-v2.json",
        _self(
            {
                "schema_version": "natural_boundary_owner_admission_census.v1",
                "status": "sealed",
                "unit_id": sealer.UNIT_ID,
                "rows": [],
            }
        ),
    )
    support_shards: list[Path] = []
    for index in range(sealer.SHARD_COUNT):
        support_shards.append(
            _write_json(
                root / "support" / f"shard-{index}.json",
                {
                    "schema_version": "natural_boundary_owner_support_completion_execution.v1",
                    "status": "completed",
                    "unit_id": sealer.UNIT_ID,
                    "checkpoint": "S",
                    "shard_index": index,
                    "plan_content_sha256": support_plan["plan_content_sha256"],
                },
            )
        )
    merge_ledger_path = _write_json(
        root / "support" / "merge-ledger.json",
        _self(
            {
                "schema_version": "natural_boundary_owner_support_completion_ledger.v1",
                "status": "completed",
                "unit_id": sealer.UNIT_ID,
                "checkpoint": "S",
                "records": [],
            },
            "content_sha256",
        ),
    )
    merge_receipt_path = _write_json(
        root / "support" / "merge-receipt.json",
        _self(
            {
                "schema_version": "natural_boundary_owner_support_completion_merge_receipt.v1",
                "status": "completed",
                "unit_id": sealer.UNIT_ID,
                "checkpoint": "S",
            }
        ),
    )
    support_inputs = {
        "plan": support_plan_path,
        "census_v2": census_v2_path,
        "shard_receipts": support_shards,
        "merge_ledger": merge_ledger_path,
        "merge_receipt": merge_receipt_path,
    }

    census_v3 = _self(
        {
            "schema_version": "natural_boundary_owner_admission_census.v3",
            "status": "sealed",
            "unit_id": sealer.UNIT_ID,
            "census_revision": "census-v3",
            "rows": [{} for _ in range(784)],
        }
    )
    census_v3_path = _write_json(root / "census-v3.json", census_v3)

    panel_path = _file(root / "panel.jsonl", "panel\n")
    cohort_path = root / "cohort.json"
    cohort_manifest_path = root / "cohort.manifest.json"
    panel = {
        "id": "human-refined-13.geo_sorted_xy",
        "path": str(panel_path),
        "sha256": sealer.sha256_file(panel_path),
    }
    inline_cohort = {"id": "natural_boundary_s_full_13_image", "frozen_arms": list(sealer.ARM_ORDER)}
    inline_operator = {"id": "natural_boundary_support_completion_v3"}
    inline_backend = {"id": "hf_fp32_sdpa_greedy"}
    events: list[dict[str, Any]] = []
    thresholds = {"minimum_event_count": 3, "minimum_image_count": 2}
    opener_contract = {
        "status": "runner_resolved",
        "resolver": "serialization_successor_runner",
        "token_name": "<|object_ref_start|>",
        "resolution_semantics": "pre_opener_natural_prefix_ends_before_object_ref_start",
        "contract_sha256": sealer.sha256_json(
            {"token_name": "<|object_ref_start|>", "resolver": "serialization_successor_runner"}
        ),
    }
    for index in range(9):
        event_id = f"gt:{5001 + index}:15"
        prefix = [50, 9, index]
        event = {
            "event_index": index,
            "event_id": event_id,
            "image_id": 5001 + index,
            "owner_refs": {
                "gt_owner_id": event_id,
                "source_panel_object_index": 15,
                "derived_panel_object_index": 15,
                "covered_owner_ids": [f"gt:{5001 + index}:14"],
                "covered_A_owner_id": f"gt:{5001 + index}:14",
            },
            "checkpoint": "S",
            "step": 2444,
            "substrate": sealer.SUBSTRATE,
            "admission": "admitted",
            "eligibility": {
                "admitted": True,
                "rule_id": "synthetic-v3",
                "thresholds": thresholds,
                "predicates": {
                    key: True for key in sealer.FROZEN_ELIGIBILITY_PREDICATES
                },
            },
            "natural_boundary": {
                "pre_opener_natural": True,
                "opener_seeded": False,
                "opener_injected": False,
                "synthetic_opener_injections": 0,
                "opener_token_id": None,
                "opener_token_contract": opener_contract,
                "prefix_token_ids": prefix,
                "prefix_sha256": sealer.sha256_json(prefix),
                "history_token_ids": prefix[1:],
                "history_sha256": sealer.sha256_json(prefix[1:]),
            },
        }
        event["event_sha256"] = sealer.sha256_json(event)
        events.append(event)
    context_owner_ids = [event["event_id"] for event in events]
    context_image_plan = {
        "cell_count": 1,
        "grid_cols": 1,
        "grid_rows": 1,
        "grid_thw": [1, 2, 2],
        "image_height": 32.0,
        "image_width": 32.0,
        "merge_size": 2,
        "merged_visual_tokens": 1,
        "observed_image_grid_thw": [1, 2, 2],
        "premerge_grid_cols": 2,
        "premerge_grid_rows": 2,
    }
    context_document = {
        "schema_version": "static_dynamic_owner_interface_cohort.v1",
        "unit_id": sealer.UNIT_ID,
        "events": [
            {
                "gt_owner_id": event["event_id"],
                "image_id": event["image_id"],
                "panel_identity": {"coco_ann_id": 100_000 + event["event_index"]},
                "geometry_by_checkpoint": {
                    "S": {"image_plan_identity": context_image_plan}
                },
            }
            for event in events
        ],
    }
    _write_json(cohort_path, context_document)
    context_manifest = {
        "schema_version": "static_dynamic_owner_interface_cohort.v1.manifest",
        "unit_id": sealer.LEGACY_CONTEXT_UNIT_ID,
        "status": "sealed",
        "cohort_path": str(cohort_path),
        "cohort_sha256": sealer.sha256_file(cohort_path),
        "event_count": len(events),
        "source_hashes": {},
        "owner_ids": context_owner_ids,
        "owner_ids_sha256": sealer.sha256_json(context_owner_ids),
    }
    _write_json(cohort_manifest_path, context_manifest)
    cohort = {
        "id": inline_cohort["id"],
        "path": str(cohort_path),
        "sha256": sealer.sha256_file(cohort_path),
        "manifest_identity_sha256": sealer.sha256_json(inline_cohort),
        "checkpoint": "S",
        "step": 2444,
        "frozen_arms": list(sealer.ARM_ORDER),
    }
    manifest = _self(
        {
            "schema_version": "s_natural_boundary_admitted_event_manifest.v3",
            "status": "sealed",
            "unit_id": sealer.UNIT_ID,
            "primary": {"checkpoint": "S", "step": 2444, "substrate": sealer.SUBSTRATE},
            "source_census": {
                "revision": "census-v3",
                "path": str(census_v3_path),
                "sha256": sealer.sha256_file(census_v3_path),
            },
            "panel": panel,
            "cohort": {**inline_cohort, "sha256": sealer.sha256_json(inline_cohort)},
            "legacy_context_cohort": {
                "status": "bound",
                "path": str(cohort_path),
                "sha256": sealer.sha256_file(cohort_path),
                "manifest_path": str(cohort_manifest_path),
                "manifest_sha256": sealer.sha256_file(cohort_manifest_path),
                "event_count": len(events),
            },
            "operator": {**inline_operator, "sha256": sealer.sha256_json(inline_operator)},
            "backend": {**inline_backend, "sha256": sealer.sha256_json(inline_backend)},
            "eligibility": {"rule_id": "synthetic-v3", "thresholds": thresholds},
            "admission_gate": {
                "minimum_event_count": 3,
                "minimum_image_count": 2,
                "status": "deferred_to_successor_runner",
            },
            "dynamic_only": {
                "status": "explicit_not_admitted_static_geometry",
                "owner_ids": [],
                "image_ids": [],
                "count": 0,
                "semantics": "synthetic empty dynamic-only set",
            },
            "arm_order": list(sealer.ARM_ORDER),
            "events": events,
            "event_count": len(events),
            "image_count": len({event["image_id"] for event in events}),
        }
    )
    manifest_path = _write_json(root / "manifest.json", manifest)
    event_refs = [
        {key: event[key] for key in ("event_index", "event_id", "image_id", "event_sha256")}
        for event in events
    ]
    shards: list[dict[str, Any]] = []
    for index in range(sealer.SHARD_COUNT):
        refs = [ref for ref in event_refs if ref["event_index"] % sealer.SHARD_COUNT == index]
        shards.append(
            {
                "shard_id": f"shard-{index:03d}",
                "shard_index": index,
                "event_indices": [ref["event_index"] for ref in refs],
                "events": refs,
                "event_count": len(refs),
                "distinct_image_count": len({ref["image_id"] for ref in refs}),
                "scalar_forward_upper_bound": len(refs) * len(sealer.ARM_ORDER) * 3 * 256,
            }
        )
    plan = {
        "schema_version": "s_natural_boundary_k_n_h_execution_plan.v1",
        "status": "planned",
        "unit_id": sealer.UNIT_ID,
        "primary": {"checkpoint": "S", "step": 2444, "substrate": sealer.SUBSTRATE},
        "manifest_sha256": sealer.sha256_file(manifest_path),
        "manifest_self_sha256": manifest["self_sha256"],
        "gate_sha256": sealer.sha256_file(gate_result_path),
        "gate_result_sha256": gate_result["result_sha256"],
        "gate_runtime_identity_sha256": gate_runtime["identity_sha256"],
        "gate_runtime_identity_raw_sha256": sealer.sha256_file(gate_runtime_path),
        "gate_scalar_forward_count": 301,
        "arm_order": list(sealer.ARM_ORDER),
        "claim_scope": {
            "execution_scope": "checkpoint_replication_candidate",
            "event_count": len(events),
            "image_count": len({event["image_id"] for event in events}),
            "minimum_checkpoint_event_count": 3,
            "minimum_checkpoint_image_count": 2,
            "checkpoint_claim_qualified": True,
            "static_direction_claim_qualified": True,
            "training_claim_qualified": False,
            "subfloor_execution_authorized": False,
        },
        "shard_count": sealer.SHARD_COUNT,
        "assignment_policy": "round_robin_event_index_mod_8",
        "event_count": len(events),
        "distinct_image_count": len({event["image_id"] for event in events}),
        "events": event_refs,
        "scalar_forward_upper_bound_per_event": len(sealer.ARM_ORDER) * 3 * 256,
        "scalar_forward_upper_bound_total": len(events) * len(sealer.ARM_ORDER) * 3 * 256,
        "gate_empirical_scalar_forward_estimate_per_event": 301,
        "gate_empirical_scalar_forward_estimate_total": len(events) * 301,
        "shards": shards,
        "no_outcome_adaptive_selection": True,
        "no_event_reorder": True,
        "no_sweep": True,
        "no_a3": True,
        "no_2x2": True,
        "no_p4": True,
    }
    plan["plan_sha256"] = sealer.sha256_json(plan)
    plan_path = _write_json(root / "execution-plan.json", plan)

    config_path = _file(root / "s-step2444.yaml", "checkpoint: S\nstep: 2444\n")
    config = {
        "id": "s-step2444-config",
        "path": str(config_path),
        "sha256": sealer.sha256_file(config_path),
        "checkpoint": "S",
        "step": 2444,
    }
    h0_root = root / "h0"
    h0_dir = h0_root / "s-step2444-h0"
    h0_dir.mkdir(parents=True)
    adapter_payload = root / "model-payload" / "adapter"
    adapter_payload.mkdir(parents=True)
    embedding_payload = root / "model-payload" / "special_token_embeddings"
    embedding_payload.mkdir(parents=True)
    base_model_dir = root / "base-model"
    base_model_dir.mkdir()
    base_names = {
        "config": "config.json",
        "model_index": "model.safetensors.index.json",
        "weight_shard_1": "model-00001-of-00002.safetensors",
        "weight_shard_2": "model-00002-of-00002.safetensors",
        "tokenizer": "tokenizer.json",
        "tokenizer_config": "tokenizer_config.json",
        "added_tokens": "added_tokens.json",
        "special_tokens_map": "special_tokens_map.json",
        "chat_template_jinja": "chat_template.jinja",
        "chat_template_json": "chat_template.json",
        "preprocessor_config": "preprocessor_config.json",
    }
    base_paths = {
        role: _file(base_model_dir / name, f"{role}\n")
        for role, name in base_names.items()
    }
    for name in (
        "generation_config.json",
        "configuration.json",
        "coord_init.json",
        "coord_tokens.json",
        "merges.txt",
        "vocab.json",
        "README.md",
        "video_preprocessor_config.json",
    ):
        _file(base_model_dir / name, f"payload {name}\n")
    base_config_sha = sealer.sha256_file(base_paths["config"])
    h0_paths = {
        "resolved_config": _write_json(
            h0_dir / "configs" / "resolved.json",
            {
                "config": {
                    "model": {"base_model": str(base_model_dir)},
                    "adapter": {"path": str(adapter_payload)},
                    "embedding_delta": {"path": str(embedding_payload)},
                }
            },
        ),
        "run_manifest": _write_json(
            h0_dir / "run_manifest.json",
            {
                "model_identity": {"base": {"path": str(base_model_dir)}},
                "adapter_identity": {
                    "adapter_path": str(adapter_payload),
                    "adapter_payload_evidence": {
                        "config_path": str(adapter_payload / "adapter_config.json"),
                        "tensor_path": str(adapter_payload / "adapter_model.safetensors"),
                    },
                },
                "backend_session": {
                    "model_identity": {
                        "embedding_delta": {
                            "identity": {
                                "delta_path": str(embedding_payload),
                                "metadata_path": str(
                                    embedding_payload / "special_token_embeddings.json"
                                ),
                            }
                        }
                    }
                },
            },
        ),
        "summary": _file(h0_dir / "summary.json", "{}\n"),
        "pred_token_trace": _file(h0_dir / "pred_token_trace.jsonl", "{}\n"),
        "image_plan": _write_json(
            h0_dir / "image_plan.jsonl",
            {
                "row_id": "coco2017_val_000000005001",
                "row_index": 0,
                "status": "ok",
                "error": None,
                "declared_width": 32,
                "declared_height": 32,
                "decoded_width": 32,
                "decoded_height": 32,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "expected_image_grid_thw": [1, 2, 2],
                "observed_image_grid_thw": [1, 2, 2],
                "raw_patch_rows": 4,
                "merged_visual_tokens": 1,
            },
        ),
        "adapter_config": _write_json(
            adapter_payload / "adapter_config.json",
            {"base_model_name_or_path": str(base_model_dir)},
        ),
        "adapter_tensor": _file(adapter_payload / "adapter_model.safetensors", "adapter"),
        "embedding_metadata": _write_json(
            embedding_payload / "special_token_embeddings.json",
            {"base_model_path": str(base_model_dir), "base_config_sha256": base_config_sha},
        ),
        "embedding_tensor": _file(
            embedding_payload / "special_token_embeddings.safetensors",
            "embedding",
        ),
    }
    h0 = {
        "id": "s-step2444-h0",
        "path": str(h0_root),
        "h0_dir": str(h0_dir),
        "checkpoint": "S",
        "step": 2444,
        "identity_files": {role: str(path) for role, path in h0_paths.items()},
        "base_model_dir": str(base_model_dir),
        "base_model_files": {role: str(path) for role, path in base_paths.items()},
    }

    sources = dict(sealer.CODE_ROLE_PATHS)
    focused_paths = list(sealer.REQUIRED_FOCUSED_TEST_PATHS)
    output = "10 passed in 0.01s\n"
    test_receipt = {
        "schema_version": "s_natural_boundary_k_n_h_focused_test_receipt.v1",
        "status": "passed",
        "producer": materializer._producer_ref(),
        "exit_code": 0,
        "command": sealer.FOCUSED_TEST_COMMAND,
        "execution_argv": materializer._focused_argv(),
        "cwd": str(sealer.REPO_ROOT),
        "passed_test_count": 10,
        "focused_tests": [
            {"path": str(path), "sha256": sealer.sha256_file(path)} for path in focused_paths
        ],
        "stdout": output,
        "stderr": "",
        "output": output,
        "output_sha256": sealer.sha256_bytes(output.encode("utf-8")),
    }
    test_receipt_path = _write_json(root / "focused-tests.json", test_receipt)
    runtime = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch_version": metadata.version("torch"),
        "transformers_version": metadata.version("transformers"),
        "backend": "hf",
        "dtype": "fp32",
        "attn_implementation": "sdpa",
        "no_training": True,
    }
    forced_math = {"enabled": True, "backend": "MATH"}
    device_policy = copy.deepcopy(sealer.DEVICE_POLICY)
    runtime_evidence_path = _write_json(
        root / "installed-runtime.json",
        {
            "schema_version": "s_natural_boundary_k_n_h_runtime_evidence.v1",
            "status": "passed",
            "producer": materializer._producer_ref(),
            "cpu_only": True,
            "gpu_used": False,
            "model_loaded": False,
            "runtime": runtime,
            "forced_math": forced_math,
            "device_policy": device_policy,
        },
    )
    shard_roots = [root / "cohort-runs" / f"shard-{index:03d}" for index in range(8)]
    final_root = root / "cohort-runs" / "merged"
    output = root / "pre-gpu" / "receipt.json"
    kwargs: dict[str, Any] = {
        "output": output,
        "unit_authority": authority,
        "gate_v3_artifacts": gate_artifacts,
        "support_inputs": support_inputs,
        "census_v3": census_v3_path,
        "manifest": manifest_path,
        "execution_plan": plan_path,
        "config": config,
        "checkpoint": {"checkpoint": "S", "step": 2444, "substrate": sealer.SUBSTRATE},
        "h0": h0,
        "panel": panel,
        "cohort": cohort,
        "cohort_manifest": cohort_manifest_path,
        "runtime": runtime,
        "runtime_evidence": runtime_evidence_path,
        "forced_math": forced_math,
        "device_policy": device_policy,
        "shard_roots": shard_roots,
        "final_merge_root": final_root,
        "source_files": sources,
        "focused_tests": focused_paths,
        "test_receipt": test_receipt_path,
    }
    paths = {
        "manifest": manifest_path,
        "execution_plan": plan_path,
        "census": census_v3_path,
        "config": config_path,
        "panel": panel_path,
        "cohort": cohort_path,
        "cohort_manifest": cohort_manifest_path,
        "h0_root": h0_root,
        "h0_dir": h0_dir,
        "base_model_dir": base_model_dir,
        "probe": sources["probe"],
        "h0_summary": h0_paths["summary"],
        "shard_roots": shard_roots,
        "final_root": final_root,
        "output": output,
    }
    return kwargs, paths


def _rebind_plan_to_manifest(
    kwargs: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    plan_path = Path(kwargs["execution_plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["manifest_sha256"] = sealer.sha256_file(kwargs["manifest"])
    plan["manifest_self_sha256"] = manifest["self_sha256"]
    plan.pop("plan_sha256")
    plan["plan_sha256"] = sealer.sha256_json(plan)
    _write_json(plan_path, plan)


def test_build_seal_validate_and_exact_full_manifest_authorization(tmp_path: Path) -> None:
    kwargs, paths = _fixture(tmp_path)
    document = sealer.build_receipt(**kwargs)
    assert document["claim_scope"]["execution_scope"] == "checkpoint_replication_candidate"
    assert document["claim_scope"]["checkpoint_claim_qualified"] is True
    assert document["event_binding"]["event_sha256s"] == document["authorized_event_sha256s"]
    assert len(document["authorized_shards"]) == 8
    assert document["authorized_shards"][0]["event_indices"] == [0, 8]
    sealer.validate_receipt(document, expected_paths={key: paths[key] for key in ("manifest", "execution_plan", "census", "config", "panel", "cohort", "h0_root", "h0_dir")}, phase="prelaunch")
    result = sealer.seal_receipt(document)
    assert result["byte_identical"] is False
    sealed = paths["output"].read_bytes()
    assert sealed == sealer.canonical_json_bytes(document) + b"\n"
    sealer.validate_receipt(document, receipt_path=paths["output"], phase="prelaunch")
    authorization = sealer.validate_shard_authorization(
        document,
        shard_id="shard-000",
        output_root=paths["shard_roots"][0],
        event_sha256s=document["authorized_shards"][0]["event_sha256s"],
        observed_cuda_visible_devices="0",
    )
    assert authorization["event_indices"] == [0, 8]
    assert authorization["physical_device"] == "0"


def test_runtime_phase_allows_two_events_in_same_existing_shard(tmp_path: Path) -> None:
    kwargs, paths = _fixture(tmp_path)
    document = sealer.build_receipt(**kwargs)
    sealer.validate_receipt(document, phase="prelaunch")
    paths["shard_roots"][0].mkdir(parents=True)
    _file(paths["shard_roots"][0] / "event-000000" / "result.json", "first event\n")
    sealer.validate_receipt(document, phase="runtime")
    with pytest.raises(sealer.PreGpuReceiptError, match="absent"):
        sealer.validate_receipt(document, phase="prelaunch")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_checkpoint", "S step-2444"),
        ("wrong_code", "exact repository owner"),
        ("wrong_output", "absent"),
        ("wrong_event_binding", "event authorization"),
    ],
)
def test_fail_closed_identity_and_output_drift(tmp_path: Path, mutation: str, message: str) -> None:
    kwargs, paths = _fixture(tmp_path)
    if mutation == "wrong_checkpoint":
        kwargs["checkpoint"] = {"checkpoint": "A3", "step": 2445}
        with pytest.raises(sealer.PreGpuReceiptError, match=message):
            sealer.build_receipt(**kwargs)
        return
    document = sealer.build_receipt(**kwargs)
    if mutation == "wrong_code":
        document = copy.deepcopy(document)
        foreign = paths["config"]
        document["source_files"]["probe"] = {
            "path": str(foreign),
            "sha256": sealer.sha256_file(foreign),
            "size_bytes": foreign.stat().st_size,
            "role": "probe",
        }
        document["code_identity"]["sha256"]["probe"] = sealer.sha256_file(foreign)
        document["self_sha256"] = sealer.document_self_sha256(document)
    elif mutation == "wrong_output":
        paths["shard_roots"][3].mkdir(parents=True)
    else:
        document = copy.deepcopy(document)
        document["authorized_event_sha256s"] = list(reversed(document["authorized_event_sha256s"]))
        document["self_sha256"] = sealer.document_self_sha256(document)
    with pytest.raises(sealer.PreGpuReceiptError, match=message):
        sealer.validate_receipt(document, phase="prelaunch")


def test_h0_source_drift_and_symlink_roots_are_rejected(tmp_path: Path) -> None:
    kwargs, paths = _fixture(tmp_path)
    document = sealer.build_receipt(**kwargs)
    paths["h0_summary"].write_text("drift\n", encoding="utf-8")
    with pytest.raises(sealer.PreGpuReceiptError, match="SHA|hash"):
        sealer.validate_receipt(document, phase="runtime")

    kwargs, paths = _fixture(tmp_path / "second")
    redirect = tmp_path / "redirect"
    redirect.mkdir()
    symlink = Path(kwargs["shard_roots"][0])
    symlink.parent.mkdir(parents=True)
    symlink.symlink_to(redirect, target_is_directory=True)
    with pytest.raises(sealer.PreGpuReceiptError, match="symlink"):
        sealer.build_receipt(**kwargs)


@pytest.mark.parametrize(
    "role_surface",
    ("h0_resolved_config", "adapter_config", "base_config"),
)
def test_loader_roles_reject_same_directory_decoy_files(
    tmp_path: Path,
    role_surface: str,
) -> None:
    kwargs, _paths = _fixture(tmp_path)
    if role_surface == "h0_resolved_config":
        selected = Path(kwargs["h0"]["identity_files"]["resolved_config"])
        decoy = _file(selected.parent / "alternate-resolved.json", selected.read_text())
        kwargs["h0"]["identity_files"]["resolved_config"] = decoy
        message = "exact loader path"
    elif role_surface == "adapter_config":
        selected = Path(kwargs["h0"]["identity_files"]["adapter_config"])
        decoy = _file(selected.parent / "alternate-adapter-config.json", selected.read_text())
        kwargs["h0"]["identity_files"]["adapter_config"] = decoy
        message = "exact selected loader path"
    else:
        selected = Path(kwargs["h0"]["base_model_files"]["config"])
        decoy = _file(selected.parent / "alternate-config.json", selected.read_text())
        kwargs["h0"]["base_model_files"]["config"] = decoy
        message = "exact loader path"
    with pytest.raises(sealer.PreGpuReceiptError, match=message):
        sealer.build_receipt(**kwargs)


def test_write_once_and_direct_file_help(tmp_path: Path) -> None:
    kwargs, paths = _fixture(tmp_path)
    document = sealer.build_receipt(**kwargs)
    assert sealer.seal_receipt(document)["byte_identical"] is False
    assert sealer.seal_receipt(document)["byte_identical"] is True
    changed = dict(document)
    changed["status"] = "changed"
    with pytest.raises(FileExistsError):
        sealer._write_once(paths["output"], changed)
    script = Path(sealer.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "build" in completed.stdout


def test_cpu_builder_cli_seals_and_validates_multi_event_receipt(tmp_path: Path) -> None:
    kwargs, paths = _fixture(tmp_path)
    h0 = kwargs["h0"]
    support = kwargs["support_inputs"]
    gate = kwargs["gate_v3_artifacts"]
    command = [
        sys.executable,
        str(Path(sealer.__file__).resolve()),
        "build",
        "--output", str(kwargs["output"]),
        "--unit-authority", str(kwargs["unit_authority"]["path"]),
        "--authority-status", str(kwargs["unit_authority"]["status"]),
        "--authority-scope", str(kwargs["unit_authority"]["scope"]),
        "--gate-result", str(gate["result"]),
        "--gate-runtime-identity", str(gate["runtime_identity"]),
        "--gate-terminal-summary", str(gate["terminal_summary"]),
        "--gate-launch-log", str(gate["launch_log"]),
        "--support-plan", str(support["plan"]),
        "--census-v2", str(support["census_v2"]),
        "--support-merge-ledger", str(support["merge_ledger"]),
        "--support-merge-receipt", str(support["merge_receipt"]),
        "--census-v3", str(kwargs["census_v3"]),
        "--manifest", str(kwargs["manifest"]),
        "--execution-plan", str(kwargs["execution_plan"]),
        "--config", str(kwargs["config"]["path"]),
        "--config-id", str(kwargs["config"]["id"]),
        "--panel", str(kwargs["panel"]["path"]),
        "--panel-id", str(kwargs["panel"]["id"]),
        "--cohort", str(kwargs["cohort"]["path"]),
        "--cohort-id", str(kwargs["cohort"]["id"]),
        "--cohort-manifest", str(kwargs["cohort_manifest"]),
        "--h0-root", str(h0["path"]),
        "--h0-dir", str(h0["h0_dir"]),
        "--h0-id", str(h0["id"]),
        "--base-model-dir", str(h0["base_model_dir"]),
        "--runtime-evidence", str(kwargs["runtime_evidence"]),
        "--focused-test-receipt", str(kwargs["test_receipt"]),
        "--final-merge-root", str(kwargs["final_merge_root"]),
    ]
    for receipt in support["shard_receipts"]:
        command.extend(("--support-shard-receipt", str(receipt)))
    for role, path in h0["identity_files"].items():
        command.extend(("--h0-file", f"{role}={path}"))
    for role, path in h0["base_model_files"].items():
        command.extend(("--base-model-file", f"{role}={path}"))
    for root in kwargs["shard_roots"]:
        command.extend(("--shard-root", str(root)))
    completed = subprocess.run(
        command,
        cwd=Path(sealer.__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    document = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert document["event_binding"]["event_count"] == 9
    assert document["authorized_shards"][0]["event_indices"] == [0, 8]
    sealer.validate_receipt(document, receipt_path=paths["output"], phase="prelaunch")
    sealer.validate_receipt(document, receipt_path=paths["output"], phase="runtime")


def test_runtime_identity_binds_full_receipt_inputs_code_and_model(tmp_path: Path) -> None:
    kwargs, paths = _fixture(tmp_path)
    document = sealer.build_receipt(**kwargs)
    sealer.seal_receipt(document)
    identity = sealer.runtime_identity_binding(
        document,
        receipt_path=paths["output"],
        shard_id="shard-007",
        observed_cuda_visible_devices="7",
    )
    assert identity["pre_gpu_receipt_sha256"] == sealer.sha256_file(paths["output"])
    assert identity["input_hashes"]["base_model_files_sha256"]
    assert identity["input_hashes"]["base_model_inventory_sha256"]
    assert identity["input_hashes"]["src_runtime_tree_sha256"]
    assert identity["device_assignment"]["physical_device"] == "7"
    assert set(identity["code_hashes"]) == set(sealer.RUNTIME_IDENTITY_CODE_ROLES)
    assert sealer.validate_runtime_identity(
        identity,
        receipt_path=paths["output"],
        receipt=document,
        shard_id="shard-007",
        observed_cuda_visible_devices="7",
    ) == identity
    tampered = copy.deepcopy(identity)
    tampered["pre_gpu_receipt_sha256"] = "0" * 64
    with pytest.raises(sealer.PreGpuReceiptError, match="runtime identity"):
        sealer.validate_runtime_identity(
            tampered,
            receipt_path=paths["output"],
            receipt=document,
            shard_id="shard-007",
            observed_cuda_visible_devices="7",
        )


@pytest.mark.parametrize("mutation", ("content", "added", "missing"))
def test_complete_base_model_inventory_rejects_unlisted_payload_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    kwargs, _paths = _fixture(tmp_path)
    document = sealer.build_receipt(**kwargs)
    assert document["primary_identity"]["h0"]["base_model_inventory"]["file_count"] == 19
    generation_config = Path(kwargs["h0"]["base_model_dir"]) / "generation_config.json"
    if mutation == "content":
        generation_config.write_text("drifted generation behavior\n", encoding="utf-8")
    elif mutation == "added":
        _file(Path(kwargs["h0"]["base_model_dir"]) / "foreign.json", "{}\n")
    else:
        generation_config.unlink()
    with pytest.raises(sealer.PreGpuReceiptError, match="base model inventory"):
        sealer.validate_receipt(document, phase="runtime")


@pytest.mark.parametrize("mutation", ("swapped_hash", "foreign_path"))
def test_src_runtime_tree_rejects_hash_or_inventory_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    kwargs, _paths = _fixture(tmp_path)
    document = sealer.build_receipt(**kwargs)
    document = copy.deepcopy(document)
    files = document["src_runtime_tree"]["files"]
    if mutation == "swapped_hash":
        files[0]["sha256"] = "0" * 64
    else:
        files.append(
            {
                "relative_path": "foreign.py",
                "sha256": "0" * 64,
                "size_bytes": 1,
            }
        )
    document["src_runtime_tree"]["file_count"] = len(files)
    document["src_runtime_tree"]["inventory_sha256"] = sealer.sha256_json(files)
    document["code_identity"]["src_runtime_tree_sha256"] = document[
        "src_runtime_tree"
    ]["inventory_sha256"]
    document["self_sha256"] = sealer.document_self_sha256(document)
    with pytest.raises(sealer.PreGpuReceiptError, match="src runtime tree inventory"):
        sealer.validate_receipt(document, phase="runtime")


def test_wrong_shard_physical_device_is_rejected_before_runtime_identity(
    tmp_path: Path,
) -> None:
    kwargs, paths = _fixture(tmp_path)
    document = sealer.build_receipt(**kwargs)
    sealer.seal_receipt(document)
    with pytest.raises(sealer.PreGpuReceiptError, match="CUDA_VISIBLE_DEVICES"):
        sealer.validate_shard_authorization(
            document,
            shard_id="shard-003",
            output_root=paths["shard_roots"][3],
            event_sha256s=document["authorized_shards"][3]["event_sha256s"],
            observed_cuda_visible_devices="4",
        )
    with pytest.raises(sealer.PreGpuReceiptError, match="CUDA_VISIBLE_DEVICES"):
        sealer.runtime_identity_binding(
            document,
            receipt_path=paths["output"],
            shard_id="shard-003",
            observed_cuda_visible_devices="4",
        )


def test_nonfinite_json_is_rejected() -> None:
    with pytest.raises(sealer.PreGpuReceiptError, match="finite"):
        sealer.canonical_json_bytes({"bad": float("nan")})


@pytest.mark.parametrize("mutation", ("status_alias", "command_alias"))
def test_focused_test_receipt_rejects_permissive_aliases(
    tmp_path: Path,
    mutation: str,
) -> None:
    kwargs, _paths = _fixture(tmp_path)
    receipt_path = Path(kwargs["test_receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "status_alias":
        receipt["status"] = "complete"
    else:
        receipt["command"] = receipt["command"].replace("python -m ", "")
    _write_json(receipt_path, receipt)
    with pytest.raises(sealer.PreGpuReceiptError, match="focused test result"):
        sealer.build_receipt(**kwargs)


def test_current_materializer_generated_receipts_round_trip_and_seal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kwargs, _paths = _fixture(tmp_path)

    def fake_run(*_args: Any, **_kwargs: Any) -> Any:
        return type("Completed", (), {"returncode": 0, "stdout": "10 passed in 0.01s\n", "stderr": ""})()

    monkeypatch.setattr(materializer.subprocess, "run", fake_run)
    result = materializer.materialize(
        focused_test_receipt=tmp_path / "generated-focused-tests.json",
        runtime_evidence=tmp_path / "generated-runtime-evidence.json",
    )
    assert result["status"] == "passed"
    kwargs["test_receipt"] = Path(result["focused_test_receipt"]["path"])
    kwargs["runtime_evidence"] = Path(result["runtime_evidence"]["path"])
    generated_runtime = json.loads(Path(kwargs["runtime_evidence"]).read_text(encoding="utf-8"))
    kwargs["runtime"] = generated_runtime["runtime"]
    kwargs["forced_math"] = generated_runtime["forced_math"]
    kwargs["device_policy"] = generated_runtime["device_policy"]
    document = sealer.build_receipt(**kwargs)
    sealer.seal_receipt(document)
    sealer.validate_receipt(document, receipt_path=kwargs["output"], phase="prelaunch")


def test_current_materializer_fails_closed_for_tamper_failed_tests_and_write_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def passing(*_args: Any, **_kwargs: Any) -> Any:
        return type("Completed", (), {"returncode": 0, "stdout": "10 passed in 0.01s\n", "stderr": ""})()

    monkeypatch.setattr(materializer.subprocess, "run", passing)
    receipt = materializer.run_focused_tests()
    target = tmp_path / "focused-tests.json"
    first = materializer._write_json_once(target, receipt)
    assert materializer._write_json_once(target, receipt)["byte_identical"] is True
    tampered = dict(receipt)
    tampered["output_sha256"] = "0" * 64
    with pytest.raises(materializer.EvidenceError, match="different bytes"):
        materializer._write_json_once(target, tampered)

    kwargs, _paths = _fixture(tmp_path / "tamper")
    _write_json(Path(kwargs["test_receipt"]), tampered)
    with pytest.raises(sealer.PreGpuReceiptError, match="output raw hash"):
        sealer.build_receipt(**kwargs)
    assert first["byte_identical"] is False

    def failing(*_args: Any, **_kwargs: Any) -> Any:
        return type("Completed", (), {"returncode": 1, "stdout": "1 failed, 8 passed\n", "stderr": "failure\n"})()

    monkeypatch.setattr(materializer.subprocess, "run", failing)
    with pytest.raises(materializer.EvidenceError, match="did not produce one passing result"):
        materializer.run_focused_tests()


def test_focused_test_receipt_accepts_interpreter_alias_but_rejects_foreign_executable(
    tmp_path: Path,
) -> None:
    kwargs, _paths = _fixture(tmp_path)
    receipt_path = Path(kwargs["test_receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    alias = tmp_path / "python-alias"
    alias.symlink_to(Path(sys.executable))
    receipt["execution_argv"][0] = str(alias)
    _write_json(receipt_path, receipt)
    sealer.build_receipt(**kwargs)

    receipt["execution_argv"][0] = "/bin/sh"
    _write_json(receipt_path, receipt)
    with pytest.raises(sealer.PreGpuReceiptError, match="executable differs"):
        sealer.build_receipt(**kwargs)


def test_frozen_manifest_and_plan_validators_are_conclusion_critical(tmp_path: Path) -> None:
    kwargs, _paths = _fixture(tmp_path)
    plan_path = Path(kwargs["execution_plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan.pop("assignment_policy")
    plan.pop("plan_sha256")
    plan["plan_sha256"] = sealer.sha256_json(plan)
    _write_json(plan_path, plan)
    with pytest.raises(sealer.PreGpuReceiptError, match="frozen planner validator"):
        sealer.build_receipt(**kwargs)

    kwargs, _paths = _fixture(tmp_path / "manifest-case")
    manifest_path = Path(kwargs["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("operator")
    manifest.pop("self_sha256")
    manifest["self_sha256"] = sealer.sha256_json(manifest)
    _write_json(manifest_path, manifest)
    _rebind_plan_to_manifest(kwargs, manifest)
    with pytest.raises(sealer.PreGpuReceiptError, match="frozen cohort validator"):
        sealer.build_receipt(**kwargs)


@pytest.mark.parametrize("mutation", ("missing", "semantic_instead_of_raw"))
def test_legacy_context_pair_requires_exact_raw_materializer_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    kwargs, _paths = _fixture(tmp_path)
    manifest_path = Path(kwargs["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        manifest.pop("legacy_context_cohort")
    else:
        cohort_document = json.loads(Path(kwargs["cohort"]["path"]).read_text())
        manifest["legacy_context_cohort"]["sha256"] = sealer.sha256_json(
            cohort_document
        )
        assert manifest["legacy_context_cohort"]["sha256"] != sealer.sha256_file(
            kwargs["cohort"]["path"]
        )
    manifest.pop("self_sha256")
    manifest["self_sha256"] = sealer.sha256_json(manifest)
    _write_json(manifest_path, manifest)
    _rebind_plan_to_manifest(kwargs, manifest)
    with pytest.raises(
        sealer.PreGpuReceiptError,
        match="legacy_context_cohort|legacy context cohort",
    ):
        sealer.build_receipt(**kwargs)


def test_manifest_image_count_is_exactly_admitted_event_images(tmp_path: Path) -> None:
    kwargs, _paths = _fixture(tmp_path)
    manifest_path = Path(kwargs["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["image_count"] += 4
    manifest.pop("self_sha256")
    manifest["self_sha256"] = sealer.sha256_json(manifest)
    _write_json(manifest_path, manifest)
    with pytest.raises(sealer.PreGpuReceiptError, match="event_count/image_count"):
        sealer.build_receipt(**kwargs)


def test_manifest_cannot_weaken_exact_admission_floor(tmp_path: Path) -> None:
    kwargs, _paths = _fixture(tmp_path)
    manifest_path = Path(kwargs["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["admission_gate"]["minimum_event_count"] = 0
    manifest["admission_gate"]["minimum_image_count"] = 0
    manifest.pop("self_sha256")
    manifest["self_sha256"] = sealer.sha256_json(manifest)
    _write_json(manifest_path, manifest)
    with pytest.raises(sealer.PreGpuReceiptError, match="replication floor identity"):
        sealer.build_receipt(**kwargs)


def test_single_event_cohort_seals_as_case_study_without_checkpoint_claim(
    tmp_path: Path,
) -> None:
    kwargs, _paths = _fixture(tmp_path)
    cohort_path = Path(kwargs["cohort"]["path"])
    companion_path = Path(kwargs["cohort_manifest"])
    context = json.loads(cohort_path.read_text(encoding="utf-8"))
    context["events"] = context["events"][:1]
    _write_json(cohort_path, context)
    companion = json.loads(companion_path.read_text(encoding="utf-8"))
    companion["cohort_sha256"] = sealer.sha256_file(cohort_path)
    companion["event_count"] = 1
    companion["owner_ids"] = [context["events"][0]["gt_owner_id"]]
    companion["owner_ids_sha256"] = sealer.sha256_json(companion["owner_ids"])
    _write_json(companion_path, companion)
    kwargs["cohort"]["sha256"] = sealer.sha256_file(cohort_path)

    manifest_path = Path(kwargs["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["events"] = manifest["events"][:1]
    manifest["event_count"] = 1
    manifest["image_count"] = 1
    manifest["legacy_context_cohort"].update(
        {
            "sha256": sealer.sha256_file(cohort_path),
            "manifest_sha256": sealer.sha256_file(companion_path),
            "event_count": 1,
        }
    )
    manifest.pop("self_sha256")
    manifest["self_sha256"] = sealer.sha256_json(manifest)
    _write_json(manifest_path, manifest)

    plan_path = Path(kwargs["execution_plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    event_ref = {
        key: manifest["events"][0][key]
        for key in ("event_index", "event_id", "image_id", "event_sha256")
    }
    plan["manifest_sha256"] = sealer.sha256_file(manifest_path)
    plan["manifest_self_sha256"] = manifest["self_sha256"]
    plan["events"] = [event_ref]
    plan["claim_scope"] = {
        "execution_scope": "case_study",
        "event_count": 1,
        "image_count": 1,
        "minimum_checkpoint_event_count": 3,
        "minimum_checkpoint_image_count": 2,
        "checkpoint_claim_qualified": False,
        "static_direction_claim_qualified": False,
        "training_claim_qualified": False,
        "subfloor_execution_authorized": True,
    }
    plan["event_count"] = 1
    plan["distinct_image_count"] = 1
    plan["scalar_forward_upper_bound_total"] = plan[
        "scalar_forward_upper_bound_per_event"
    ]
    plan["gate_empirical_scalar_forward_estimate_total"] = plan[
        "gate_empirical_scalar_forward_estimate_per_event"
    ]
    for index, shard in enumerate(plan["shards"]):
        refs = [event_ref] if index == 0 else []
        shard["event_indices"] = [ref["event_index"] for ref in refs]
        shard["events"] = refs
        shard["event_count"] = len(refs)
        shard["distinct_image_count"] = len(refs)
        shard["scalar_forward_upper_bound"] = (
            len(refs) * plan["scalar_forward_upper_bound_per_event"]
        )
    plan.pop("plan_sha256")
    plan["plan_sha256"] = sealer.sha256_json(plan)
    _write_json(plan_path, plan)

    document = sealer.build_receipt(**kwargs)
    assert document["claim_scope"] == {
        "execution_scope": "case_study",
        "event_count": 1,
        "image_count": 1,
        "minimum_checkpoint_event_count": 3,
        "minimum_checkpoint_image_count": 2,
        "checkpoint_claim_qualified": False,
        "static_direction_claim_qualified": False,
        "training_claim_qualified": False,
        "subfloor_execution_authorized": True,
    }
    sealer.validate_receipt(document, phase="prelaunch")


@pytest.mark.parametrize("mutation", ("false_predicate", "missing_derived_index"))
def test_manifest_cannot_self_authorize_ineligible_or_unowned_event(
    tmp_path: Path,
    mutation: str,
) -> None:
    kwargs, _paths = _fixture(tmp_path)
    manifest_path = Path(kwargs["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event = manifest["events"][0]
    if mutation == "false_predicate":
        event["eligibility"]["predicates"]["geometry_launch_eligible"] = False
    else:
        event["owner_refs"].pop("derived_panel_object_index")
    event.pop("event_sha256")
    event["event_sha256"] = sealer.sha256_json(event)
    manifest.pop("self_sha256")
    manifest["self_sha256"] = sealer.sha256_json(manifest)
    _write_json(manifest_path, manifest)
    with pytest.raises(sealer.PreGpuReceiptError, match="admission identity/predicates"):
        sealer.build_receipt(**kwargs)


def test_legacy_context_requires_panel_coco_annotation_identity(tmp_path: Path) -> None:
    kwargs, _paths = _fixture(tmp_path)
    cohort_path = Path(kwargs["cohort"]["path"])
    companion_path = Path(kwargs["cohort_manifest"])
    cohort_document = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_document["events"][0]["panel_identity"].pop("coco_ann_id")
    _write_json(cohort_path, cohort_document)
    companion = json.loads(companion_path.read_text(encoding="utf-8"))
    companion["cohort_sha256"] = sealer.sha256_file(cohort_path)
    _write_json(companion_path, companion)
    manifest = json.loads(Path(kwargs["manifest"]).read_text(encoding="utf-8"))
    manifest["legacy_context_cohort"]["sha256"] = sealer.sha256_file(cohort_path)
    manifest["legacy_context_cohort"]["manifest_sha256"] = sealer.sha256_file(
        companion_path
    )
    cohort_identity = {
        **kwargs["cohort"],
        "sha256": sealer.sha256_file(cohort_path),
    }
    with pytest.raises(sealer.PreGpuReceiptError, match="coco_ann_id"):
        sealer._validate_legacy_context_binding(
            manifest,
            cohort_identity,
            companion_path,
        )


def test_legacy_context_accepts_stable_negative_human_refined_annotation_id(
    tmp_path: Path,
) -> None:
    kwargs, _paths = _fixture(tmp_path)
    cohort_path = Path(kwargs["cohort"]["path"])
    companion_path = Path(kwargs["cohort_manifest"])
    cohort_document = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_document["events"][0]["panel_identity"]["coco_ann_id"] = -73
    _write_json(cohort_path, cohort_document)
    companion = json.loads(companion_path.read_text(encoding="utf-8"))
    companion["cohort_sha256"] = sealer.sha256_file(cohort_path)
    _write_json(companion_path, companion)
    manifest = json.loads(Path(kwargs["manifest"]).read_text(encoding="utf-8"))
    manifest["legacy_context_cohort"]["sha256"] = sealer.sha256_file(cohort_path)
    manifest["legacy_context_cohort"]["manifest_sha256"] = sealer.sha256_file(
        companion_path
    )
    cohort_identity = {
        **kwargs["cohort"],
        "sha256": sealer.sha256_file(cohort_path),
    }

    binding = sealer._validate_legacy_context_binding(
        manifest,
        cohort_identity,
        companion_path,
    )

    assert binding["event_count"] == len(cohort_document["events"])


def test_actual_gate_v3_is_accepted_as_no_training_evidence() -> None:
    unit_root = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-06-natural-boundary-routing-history-replication"
    )
    gate_root = unit_root / "s-gt5001-live-gate-v3"
    refs, documents = sealer._validate_gate_artifacts(
        {
            "result": gate_root / "result.json",
            "runtime_identity": gate_root / "runtime_identity.json",
            "terminal_summary": gate_root / "terminal_summary.json",
            "launch_log": unit_root / "s-gt5001-live-gate-v3.launch.log",
        }
    )
    assert documents["result"]["gpu_launch_authorized"] is False
    assert documents["result"]["no_training"] is True
    assert refs["result"]["sha256"] == "81b6e9557fd7f0eed54b52ea2c34bee1d62056f4f2459d7f24bfee1c4fcef309"
