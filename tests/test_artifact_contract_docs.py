from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_artifact_contract_docs_freeze_rank0_and_stage2_eval_surfaces() -> None:
    artifacts = (REPO_ROOT / "docs" / "ARTIFACTS.md").read_text(encoding="utf-8")

    for artifact_name in (
        "resolved_config.json",
        "runtime_env.json",
        "effective_runtime.json",
        "pipeline_manifest.json",
        "experiment_manifest.json",
        "run_metadata.json",
        "train_data_provenance.json",
        "eval_data_provenance.json",
        "config_source.yaml",
        "base_config_source.yaml",
    ):
        assert artifact_name in artifacts

    assert "rollout_matching.eval_detection.materialize_artifacts: true" in artifacts
    assert "Artifact/Provenance Freeze" in artifacts
    assert "Stage-2 Policy Provenance" in artifacts
    assert "Diagnostic Compatibility Freeze" in artifacts
    assert "Rank-0 Stage-2 two-channel" in artifacts
    assert "not yet written by all rank-0 manifests" not in artifacts

    for policy_surface in (
        "stage2_policy_provenance.schema_version",
        "stage2_policy_provenance.trainer_variant",
        "stage2_policy_provenance.assignment_strategy",
        "stage2_policy_provenance.duplicate_filter_strategy",
        "stage2_policy_provenance.object_ordering_policy",
        "stage2_policy_provenance.assignment_iou_threshold_effective",
        "stage2_policy_provenance.object_ordering_strategy_id",
        "stage2_policy_provenance.rollout_template_family",
        "stage2_policy_provenance.rollout_decode_policy",
        "stage2_policy_provenance.invalid_rollout_policy",
        "legacy_hungarian_mask_iou",
        "legacy_tail_append",
        "src/trainers/rollout_matching/matching.py::hungarian_match_maskiou",
        "src/trainers/stage2_two_channel/target_builder.py::_apply_channel_b_duplicate_control",
        "stage2_ab.channel_b.insertion_order",
    ):
        assert policy_surface in artifacts

    assert "Blocking migration gap" not in artifacts

    for diagnostic_surface in (
        "monitor_dumps/",
        "prepare_failures/",
        "raw_rollouts.jsonl",
        "pred_token_trace.jsonl",
        "gt_vs_pred_guarded.jsonl",
        "duplicate/EOS diagnostic probes",
    ):
        assert diagnostic_surface in artifacts


def test_stage2_ab_spec_allows_channel_b_insertion_order_key() -> None:
    spec = (
        REPO_ROOT / "openspec" / "specs" / "stage2-ab-training" / "spec.md"
    ).read_text(encoding="utf-8")
    allowed_key_section = spec.split(
        "- `stage2_ab.channel_b` MUST accept only:",
        maxsplit=1,
    )[1].split(
        "- `stage2_ab.channel_b.duplicate_control` MUST be a typed mapping",
        maxsplit=1,
    )[0]

    assert "- `insertion_order`" in allowed_key_section
    assert "stage2_ab.channel_b.insertion_order: tail_append" in spec
    assert "MUST remain the default" in spec


def test_training_decision_export_matches_stage2_shadow_ordering_contract() -> None:
    decisions = (
        REPO_ROOT
        / "progress"
        / "explorations"
        / "2026-05-15_training_infrastructure_architecture_decisions.md"
    ).read_text(encoding="utf-8")

    assert "### Decision 46: Duplicate Filtering Runs Before Assignment" in decisions
    assert "Assignment Runs Before Duplicate Filtering" not in decisions
    assert (
        "RolloutViews\n"
        "  -> DuplicateFilter\n"
        "  -> DuplicateFilterResult\n"
        "  -> accepted survivors\n"
        "  -> AssignmentStrategy"
    ) in decisions
    assert "- `tail_append`\n  - current default compatibility mode;" in decisions
    assert "future canonical default only after an explicit" in decisions
    assert "assignment-first Stage-2 duplicate filtering" not in decisions
    assert "  - canonical stable ordering." not in decisions


def test_superpowers_architecture_spec_matches_stage2_shadow_ordering_contract() -> None:
    spec = (
        REPO_ROOT
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-05-15-unified-training-infrastructure-architecture-design.md"
    ).read_text(encoding="utf-8")

    assert (
        "RolloutViews\n"
        "  -> DuplicateFilter\n"
        "  -> DuplicateFilterResult\n"
        "  -> accepted survivors\n"
        "  -> AssignmentStrategy\n"
        "  -> AssignmentResult"
    ) in spec
    assert "assignment-aware duplicate filtering" not in spec
    assert "AssignmentStrategy\n  -> AssignmentResult\n  -> DuplicateFilter" not in spec


def test_catalog_unified_shadow_surfaces_share_closed_domains() -> None:
    catalog = yaml.safe_load(
        (REPO_ROOT / "docs" / "catalog.yaml").read_text(encoding="utf-8")
    )
    training_surfaces = catalog["config_surfaces"]["training"]
    shadow_surfaces = [
        surface
        for surface in training_surfaces
        if surface["id"].startswith("unified_training_shadow_")
    ]
    expected_domains = [
        "run",
        "surface",
        "data",
        "template",
        "supervision",
        "objectives",
        "observability",
        "artifacts",
        "runtime",
    ]

    assert {surface["surface_id"] for surface in shadow_surfaces} == {
        "stage1_json_ce",
        "stage1_compact_trie_ce",
        "stage2_two_channel",
    }
    assert all(surface["domains"] == expected_domains for surface in shadow_surfaces)


def test_catalog_does_not_advertise_removed_a2e_training_surface() -> None:
    catalog = yaml.safe_load(
        (REPO_ROOT / "docs" / "catalog.yaml").read_text(encoding="utf-8")
    )
    progress = yaml.safe_load(
        (REPO_ROOT / "progress" / "index.yaml").read_text(encoding="utf-8")
    )

    notes_by_path = {note["path"]: note for note in progress["notes"]}
    a2e_note = notes_by_path[
        "progress/diagnostics/2026-05-14_a2_eos_"
        "loosen_ablation.md"
    ]
    assert a2e_note["status"] == "concluded-negative"

    training_surfaces = {
        surface["id"]: surface for surface in catalog["config_surfaces"]["training"]
    }
    assert (
        "stage1_latest_compact_detection_a2_eos_"
        "loosen_ablation"
        not in training_surfaces
    )


def test_progress_routers_do_not_promote_concluded_negative_a2e_as_active() -> None:
    catalog = yaml.safe_load(
        (REPO_ROOT / "docs" / "catalog.yaml").read_text(encoding="utf-8")
    )
    progress = yaml.safe_load(
        (REPO_ROOT / "progress" / "index.yaml").read_text(encoding="utf-8")
    )
    diagnostics_readme = (
        REPO_ROOT / "progress" / "diagnostics" / "README.md"
    ).read_text(encoding="utf-8")

    notes_by_path = {note["path"]: note for note in progress["notes"]}
    a2e_note = notes_by_path[
        "progress/diagnostics/2026-05-14_a2_eos_"
        "loosen_ablation.md"
    ]
    assert a2e_note["status"] == "concluded-negative"

    training_surfaces = {
        surface["id"]: surface for surface in catalog["config_surfaces"]["training"]
    }
    assert (
        "stage1_latest_compact_detection_a2_eos_"
        "loosen_ablation"
        not in training_surfaces
    )

    forbidden_active_phrases = (
        "Active A2 EOS-Loosen",
        "Active A2E",
        "running A2E",
        "current A2E recommendation",
    )
    for phrase in forbidden_active_phrases:
        assert phrase not in diagnostics_readme

    assert "A2 EOS-Loosen Clean Ablation" in diagnostics_readme
    assert "concluded negative" in diagnostics_readme.lower()
