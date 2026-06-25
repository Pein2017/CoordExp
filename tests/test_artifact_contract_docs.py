from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


_MARKDOWN_LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
_REPO_DOC_HANDLE_PATTERN = re.compile(
    r"(?<![\w./-])((?:docs|research)/(?:[A-Za-z0-9_./@+-]+))"
)


def _assert_local_markdown_links_are_tracked(markdown_path: Path) -> None:
    content = markdown_path.read_text(encoding="utf-8")
    _assert_repo_relative_targets_are_tracked(
        markdown_path,
        _local_markdown_targets(content),
    )
    _assert_repo_relative_targets_are_tracked(
        markdown_path,
        _repo_doc_handles(content),
    )


def _local_markdown_targets(content: str) -> list[str]:
    targets: list[str] = []
    for raw_target in _MARKDOWN_LINK_PATTERN.findall(content):
        target = raw_target.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        targets.append(target)
    return targets


def _repo_doc_handles(content: str) -> list[str]:
    targets: list[str] = []
    for raw_target in _REPO_DOC_HANDLE_PATTERN.findall(content):
        target = raw_target.rstrip(".,;:")
        if target:
            targets.append(target)
    return targets


def _assert_repo_relative_targets_are_tracked(
    markdown_path: Path,
    targets: list[str],
) -> None:
    base_dir = markdown_path.parent
    for target in targets:
        resolved = (base_dir / target).resolve()
        if target.startswith(("docs/", "research/")):
            resolved = (REPO_ROOT / target).resolve()
        try:
            relative = resolved.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise AssertionError(
                f"{markdown_path.relative_to(REPO_ROOT)} references outside repo: "
                f"{target}"
            ) from exc
        assert resolved.exists(), (
            f"{markdown_path.relative_to(REPO_ROOT)} references missing path: "
            f"{target}"
        )
        if resolved.is_dir():
            tracked = subprocess.run(
                ["git", "ls-files", str(relative)],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            assert tracked.stdout.strip(), (
                f"{markdown_path.relative_to(REPO_ROOT)} references directory "
                f"without tracked files: {target}"
            )
        else:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(relative)],
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            assert tracked.returncode == 0, (
                f"{markdown_path.relative_to(REPO_ROOT)} references untracked file: "
                f"{target}"
            )


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
    assert "Rank-0 Stage-2 rollout-correction" in artifacts
    assert "not yet written by all rank-0 manifests" not in artifacts

    for policy_surface in (
        "stage2_policy_provenance.schema_version",
        "stage2_policy_provenance.pipeline.id",
        "stage2_policy_provenance.assignment_strategy",
        "stage2_policy_provenance.duplicate_filter_strategy",
        "stage2_policy_provenance.object_ordering_policy",
        "stage2_policy_provenance.assignment_iou_threshold_effective",
        "stage2_policy_provenance.object_ordering_strategy_id",
        "stage2_policy_provenance.rollout_template_family",
        "stage2_policy_provenance.invalid_rollout_policy",
        "stage2_policy_provenance.fallback_loss_weight",
        "src/training/stage2/assignment.py::GreedyIoUAssignment",
        "stage2_rollout_correction.correction.duplicate_control",
        "stage2_rollout_correction.correction.insertion_order",
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


def test_coverage_ledger_launch_prep_docs_freeze_smoke_artifacts_and_metrics() -> None:
    stage1 = (REPO_ROOT / "docs" / "training" / "STAGE1_OBJECTIVE.md").read_text(
        encoding="utf-8"
    )
    metrics = (REPO_ROOT / "docs" / "training" / "METRICS.md").read_text(
        encoding="utf-8"
    )
    artifacts = (REPO_ROOT / "docs" / "ARTIFACTS.md").read_text(encoding="utf-8")
    design = (
        REPO_ROOT
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-06-23-coverage-ledger-auxiliary-loss-design.md"
    ).read_text(encoding="utf-8")
    research_index = (
        REPO_ROOT / "research" / "ideas" / "ledger-auxiliary-loss" / "index.md"
    ).read_text(encoding="utf-8")

    assert "closed-wrapper hard-SFT ledger smoke route" in stage1
    assert (
        "configs/stage1/detection_teacher_forcing/smoke/"
        "coverage_ledger_closed_hard_sft_128_baseline.yaml"
    ) in stage1
    assert (
        "configs/stage1/detection_teacher_forcing/smoke/"
        "coverage_ledger_closed_hard_sft_128.yaml"
    ) in stage1
    assert "older positive-only anchor semantics" in stage1
    assert "one-vs-all row-object" in stage1
    assert "binding change" in stage1
    assert "Do not launch production training" in stage1
    assert "coverage_ledger_closed_hard_sft.yaml" in stage1

    for metric_key in (
        "teacher_forcing/loss/coverage_ledger_auxiliary/contribution",
        "teacher_forcing/ledger/coverage_ledger_auxiliary_pair_normalized",
        "teacher_forcing/ledger/coverage_bce",
        "teacher_forcing/ledger/row_object_binding_bce",
        "teacher_forcing/ledger/coverage_auc",
        "teacher_forcing/ledger/coverage_accuracy",
        "teacher_forcing/ledger/row_object_binding_auc",
        "teacher_forcing/ledger/row_object_binding_accuracy",
        "teacher_forcing/ledger/coverage_state_count",
        "teacher_forcing/ledger/coverage_pair_count",
        "teacher_forcing/ledger/object_count",
        "teacher_forcing/ledger/row_object_binding_pair_count",
    ):
        assert metric_key in metrics

    assert "Reducer summary for coverage-ledger keys" in metrics
    assert "exact weighted auxiliary scalar" in metrics
    assert "weighted_mean" in metrics
    assert "ratio" in metrics
    assert "sum" in metrics
    assert "diagnostic_only=true" in metrics
    assert "diagnostic_only=false" in metrics

    for artifact_name in (
        "ledger/selected_samples.json",
        "ledger/alignment_debug.jsonl",
        "ledger/overlays/",
    ):
        assert artifact_name in artifacts

    assert "Smoke launch status: a prior adapter-save smoke pair completed" in artifacts
    assert "source_jsonl_repo_path" in design
    assert "source_jsonl_resolved_path" in design
    assert "debug.train_sample_selection" in artifacts
    assert "must not already contain files" in artifacts
    assert "all-128 `ledger/alignment_debug.jsonl`" in artifacts
    assert "16 overlays" in artifacts
    assert "baseline and ledger config diff remains allowlisted" in artifacts
    assert "ledger metrics appear in train logs" in artifacts
    assert "training/objectives/coverage_ledger/*" in artifacts

    assert (
        "../../../docs/superpowers/plans/"
        "2026-06-23-coverage-ledger-auxiliary-loss.md"
    ) in research_index
    assert "OpenSpec is deferred" in research_index
    assert "not current stable behavior" in research_index
    for markdown_path in (
        REPO_ROOT / "research" / "ideas" / "ledger-auxiliary-loss" / "index.md",
        REPO_ROOT / "research" / "ideas" / "ledger-auxiliary-loss" / "overview.md",
        REPO_ROOT / "research" / "ideas" / "ledger-auxiliary-loss" / "draft.md",
        REPO_ROOT / "research" / "ideas" / "ledger-auxiliary-loss" / "discussion.md",
        REPO_ROOT
        / "docs"
        / "superpowers"
        / "plans"
        / "2026-06-23-coverage-ledger-auxiliary-loss.md",
    ):
        _assert_local_markdown_links_are_tracked(markdown_path)


def test_stage2_rollout_correction_spec_rejects_removed_scheduler_and_channel_keys() -> None:
    spec = (
        REPO_ROOT / "openspec" / "specs" / "stage2-rollout-correction" / "spec.md"
    ).read_text(encoding="utf-8")

    assert "`stage2_rollout_correction.schedule`" in spec
    assert "`stage2_rollout_correction.b_ratio`" in spec
    assert "`stage2_rollout_correction.channel_b`" in spec
    assert "`stage2_rollout_correction.pipeline.objective[].channels`" in spec
    assert "`_stage2_ab_channel`" in spec


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
        / "history"
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


def test_catalog_unified_pipeline_routes_share_closed_domains() -> None:
    catalog = yaml.safe_load(
        (REPO_ROOT / "docs" / "catalog.yaml").read_text(encoding="utf-8")
    )
    training_routes = catalog["config_surfaces"]["training"]
    pipeline_routes = [
        route
        for route in training_routes
        if route["id"].startswith("unified_training_pipeline_")
    ]
    expected_domains = [
        "run",
        "pipeline",
        "data",
        "template",
        "supervision",
        "objectives",
        "observability",
        "artifacts",
        "runtime",
    ]

    assert {route["pipeline_id"] for route in pipeline_routes} == {
        "stage2_rollout_correction",
    }
    assert all(route["domains"] == expected_domains for route in pipeline_routes)


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
