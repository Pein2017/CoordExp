from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import pytest
import yaml

from src.config.loader import ConfigLoader
from src.config.schema import PromptOverrides, TrainingConfig
from src.trainers.rollout_correction.objective_runner import (
    build_stage2_core_loss_logs,
)
from src.trainers.teacher_forcing.module_registry import OBJECTIVE_MODULE_CATALOG


REMOVED_DUPLICATE_BURST_OBJECTIVE = "loss_duplicate_burst_unlikelihood"
REMOVED_DUPLICATE_BURST_LIVE_KEYS = {
    "train/optimization/loss_duplicate_burst_unlikelihood",
    "loss/rollout_correction_text/duplicate_burst_unlikelihood",
    "loss_duplicate_burst_unlikelihood_contrib",
}
REMOVED_ADJACENT_REPULSION_CONFIG_KEYS = {
    "adjacent_repulsion_weight",
    "adjacent_repulsion_filter_mode",
    "adjacent_repulsion_margin_ratio",
    "adjacent_repulsion_copy_margin",
}
REMOVED_ADJACENT_REPULSION_LIVE_KEYS = {
    "coord_softce_w1/adjacent_repulsion",
    "coord_diag/adjacent_repulsion",
    "coord_diag/adjacent_repulsion_pair_count",
    "coord_diag/adjacent_repulsion_applied_count",
    "coord_diag/adjacent_repulsion_copy_score_mean",
    "loss/adjacent_repulsion",
    "loss/rollout_correction_coord/adjacent_repulsion",
    "adjacent_repulsion_contrib",
}
ACTIVE_OPENSPEC_CHANGE_EXCLUSIONS_FOR_REMOVED_MECHANISMS: set[str] = set()
TASK_1C_CONTINUE_OVER_EOS_TERMS = {
    "continue_over_eos_margin",
    "continue_over_eos_weight",
    "birth_first",
}
TASK_1C_EOS_LOOSEN_TERMS = {
    "a4-eos",
    "a4_eos",
    "eos trust",
    "eos-trust",
    "eos_loosen",
    "eos_trust",
    "missing_label_prior_weighted_ce",
    "eos_trust_weight",
    "eos_weighted_loss",
}
TASK_1C_BOUNDARY_FORCING_TERMS = {
    "separator_continue_weight",
    "eos_stop_weight",
}
TASK_1C_STOP_GATE_TERMS = {
    "stop_signal_damping",
    "stop_signal_ce",
    "stop_signal/",
    "stop_gate",
}
TASK_1C_ACTIVE_SCAN_ROOTS = (
    "configs",
    "src",
    "tests",
    "docs",
    "openspec/specs",
    "openspec/changes",
)
TASK_1C_ACTIVE_SCAN_EXCLUDED_PREFIXES = (
    "src/analysis/",
    "docs/superpowers/",
    "openspec/changes/archive/",
    "progress/",
)
TASK_1C_ALLOWED_DECISION_EVIDENCE_PATHS = {
    "progress/diagnostics/2026-04-22_stage2_birth_first_channel_b_decision_study.md": {
        "catalog_status": "decision-evidence",
        "terms": {"birth_first"},
    },
}
PROHIBITED_ADJACENT_REPULSION_LIVE_SUPPORT_CLAIMS = (
    "shall support adjacent",
    "shall support an optional adjacent",
    "shall implement adjacent",
    "must support adjacent",
    "when adjacent repulsion is enabled",
    "when stage-1 adjacent repulsion",
    "when stage-2 adjacent repulsion",
    "adjacent repulsion contributes",
)

RETAINED_DUPLICATE_DIAGNOSTIC_KEYS = {
    "stage2_rollout_correction/correction/dup/N_clusters_total",
    "stage2_rollout_correction/correction/dup/N_clusters_suppressed",
    "stage2_rollout_correction/correction/dup/N_objects_suppressed",
    "stage2_rollout_correction/correction/dup/N_duplicate_control_first_divergence_boundaries",
    "stage2_rollout_correction/correction/dup/N_duplicate_control_first_divergence_skipped_no_divergence",
}

ALLOWED_DOC_REMOVAL_CONTEXT = (
    "removed",
    "retired",
    "historical",
    "compatibility",
    "diagnostic",
    "diagnostics",
    "no longer",
    "not part of active",
    "not a live",
    "must reject",
    "rejects",
)

PROHIBITED_STALE_DUPLICATE_BURST_CLAIMS = (
    "duplicate-ul supervision",
    "duplicate ul must target",
    "duplicate-ul positives",
    "dead-anchor suppression objective module",
    "dead_anchor_suppression_targets",
    "bad-token suppression targets",
    "loss-consumed suppression targets",
    "duplicate-burst unlikelihood remains",
    "may contribute duplicate-burst unlikelihood",
    "duplicate-ul supervision must",
    "ul payload shape",
    "shall define duplicate unlikelihood as a boundary-local objective",
    "represented only through duplicate-ul supervision",
    "duplicate-unlikelihood target",
    "first-divergence targets",
    "diagnostic target",
    "target list",
)

PROHIBITED_STALE_DUPLICATE_METRIC_KEYS = (
    "stage2_rollout_correction/correction/dup/N_ul_boundaries",
    "stage2_rollout_correction/correction/dup/N_duplicate_burst_unlikelihood_skipped_no_divergence",
    "diag/duplicate_burst/",
)

REMOVED_FUSION_PATHS = (
    "configs/fusion",
    "src/datasets/fusion.py",
    "src/datasets/fusion_types.py",
    "src/datasets/unified_fusion_dataset.py",
    "src/callbacks/fusion_epoch.py",
    "tests/test_fusion_config.py",
)
REMOVED_FUSION_MODULES = (
    "src.datasets.fusion",
    "src.datasets.fusion_types",
    "src.datasets.unified_fusion_dataset",
    "src.callbacks.fusion_epoch",
)
REMOVED_PUBLIC_STAGE2_MODULE_PATHS = (
    "src/trainers/rollout_matching_sft.py",
    "src/trainers/stage2_ab_training.py",
    "src/trainers/stage2_ab",
)
REMOVED_PUBLIC_STAGE2_MODULES = (
    "src.trainers.rollout_matching_sft",
    "src.trainers.stage2_ab_training",
    "src.trainers.stage2_ab",
)

STALE_DUPLICATE_BURST_ANALYSIS_ARTIFACT_FIELD = (
    "duplicate_burst_unlikelihood_boundary_count"
)
CANONICAL_DUPLICATE_CONTROL_ANALYSIS_ARTIFACT_FIELD = (
    "duplicate_control_first_divergence_boundary_count"
)

ALLOWED_ACTIVE_OPENSPEC_CONTEXT = (
    "removed",
    "retired",
    "superseded",
    "deferred",
    "diagnostic",
    "diagnostics",
    "no longer",
    "not a live",
    "must reject",
    "rejects",
    "must omit",
    "unavailable",
)


def _iter_task_1c_active_text_lines(repo_root: Path):
    guard_path = Path(__file__).resolve()

    for root_name in TASK_1C_ACTIVE_SCAN_ROOTS:
        root = repo_root / root_name
        if not root.exists():
            continue

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path == guard_path:
                continue

            rel_path = path.relative_to(repo_root).as_posix()
            if rel_path.endswith((".pyc", ".png", ".jpg", ".jpeg", ".webp")):
                continue
            if any(
                rel_path.startswith(prefix)
                for prefix in TASK_1C_ACTIVE_SCAN_EXCLUDED_PREFIXES
            ):
                continue

            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue

            for lineno, line in enumerate(lines, 1):
                yield rel_path, lineno, line


def _assert_task_1c_terms_absent(terms: set[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    decision_evidence_statuses = _catalog_decision_evidence_statuses(repo_root)

    offenders: list[str] = []
    for rel_path, lineno, line in _iter_task_1c_active_text_lines(repo_root):
        normalized = line.lower()
        for term in sorted(terms):
            if term.lower() in normalized:
                if _is_allowed_task_1c_decision_evidence(
                    rel_path=rel_path,
                    line=line,
                    term=term,
                    decision_evidence_statuses=decision_evidence_statuses,
                ):
                    continue
                offenders.append(f"{rel_path}:{lineno}:{line.strip()}")

    assert offenders == []


def _catalog_decision_evidence_statuses(repo_root: Path) -> dict[str, str]:
    catalog_path = repo_root / "docs" / "catalog.yaml"
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))

    statuses: dict[str, str] = {}
    for group in catalog.get("progress", {}).values():
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            status = item.get("status")
            if isinstance(path, str) and isinstance(status, str):
                statuses[path] = status
    return statuses


def _is_allowed_task_1c_decision_evidence(
    *,
    rel_path: str,
    line: str,
    term: str,
    decision_evidence_statuses: dict[str, str],
) -> bool:
    if rel_path != "docs/catalog.yaml":
        return False

    normalized_term = term.lower()
    for evidence_path, policy in TASK_1C_ALLOWED_DECISION_EVIDENCE_PATHS.items():
        if normalized_term not in policy["terms"]:
            continue
        if evidence_path not in line:
            continue
        return decision_evidence_statuses.get(evidence_path) == policy["catalog_status"]

    return False


def _canonical_live_stage2_objective() -> list[dict]:
    return [
        {
            "name": "residual_set_correction",
            "enabled": True,
            "weight": 1.0,
            "application": {"preset": "rollout_self_prefix"},
            "config": {
                "expected_num_rollouts": 4,
                "base_seed": 17,
                "lambda_type": 1.0,
                "lambda_inner": 1.0,
                "fallback_loss_weight": 1.0,
                "lambda_ul_promoted": 0.5,
                "label_conflict_weight": 0.25,
                "commit_iou_threshold": 0.75,
                "duplicate_burst_iou_threshold": 0.95,
                "ul_cluster_iou_threshold": 0.9,
                "ul_gray_iou_low": 0.30,
                "ul_consensus_ratio": 1.0,
                "min_ul_valid_rollouts": 4,
                "clean_gt_sft_mix": 0,
                "strict_builder_invariants": True,
            },
        },
    ]


def _stage2_training_payload(*, objective: list[dict]) -> dict:
    return {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
            "user_prompt": "{bbox}",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_rollout_correction",
        },
        "training": {"per_device_train_batch_size": 1, "effective_batch_size": 1},
        "rollout_matching": {
            "rollout_backend": "hf",
            "rollout_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
        },
        "stage2_rollout_correction": {
            "pipeline": {
                "objective": objective,
                "diagnostics": [],
            },
            "correction": {},
        },
    }


def _duplicate_burst_objective_spec() -> dict:
    return {
        "name": REMOVED_DUPLICATE_BURST_OBJECTIVE,
        "enabled": True,
        "weight": 1.0,
        "application": {"preset": "rollout_self_prefix"},
        "config": {},
    }


def test_duplicate_burst_unlikelihood_is_not_in_objective_catalog() -> None:
    assert REMOVED_DUPLICATE_BURST_OBJECTIVE not in OBJECTIVE_MODULE_CATALOG


def test_task_1c_continue_over_eos_and_birth_first_terms_are_absent() -> None:
    _assert_task_1c_terms_absent(TASK_1C_CONTINUE_OVER_EOS_TERMS)


def test_task_1c_eos_loosen_terms_are_absent_from_live_training_surfaces() -> None:
    _assert_task_1c_terms_absent(TASK_1C_EOS_LOOSEN_TERMS)


def test_task_1c_boundary_forcing_terms_are_absent_from_live_training_surfaces() -> None:
    _assert_task_1c_terms_absent(TASK_1C_BOUNDARY_FORCING_TERMS)


def test_task_1c_stop_gate_terms_are_absent_from_live_training_surfaces() -> None:
    _assert_task_1c_terms_absent(TASK_1C_STOP_GATE_TERMS)


def test_removed_fusion_training_surface_files_and_modules_are_absent() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for rel_path in REMOVED_FUSION_PATHS:
        assert not (repo_root / rel_path).exists(), rel_path
    for module_name in REMOVED_FUSION_MODULES:
        assert importlib.util.find_spec(module_name) is None, module_name


def test_removed_public_stage2_trainer_modules_are_absent() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for rel_path in REMOVED_PUBLIC_STAGE2_MODULE_PATHS:
        assert not (repo_root / rel_path).exists(), rel_path
    for module_name in REMOVED_PUBLIC_STAGE2_MODULES:
        assert importlib.util.find_spec(module_name) is None, module_name


@pytest.mark.parametrize("module_name", ["bbox_geo", "bbox_size_aux", "coord_reg"])
def test_removed_geometry_modules_are_not_in_objective_catalog(module_name: str) -> None:
    assert module_name not in OBJECTIVE_MODULE_CATALOG


@pytest.mark.parametrize(
    "rel_path",
    [
        "src/trainers/teacher_forcing/modules/bbox_geo.py",
        "src/trainers/teacher_forcing/modules/bbox_size_aux.py",
        "src/trainers/teacher_forcing/modules/coord_reg.py",
        "src/trainers/teacher_forcing/modules/coord_diag.py",
        "src/trainers/metrics/bbox_losses.py",
        "src/trainers/losses/bbox_geo.py",
        "src/trainers/losses/bbox_size_aux.py",
    ],
)
def test_removed_geometry_loss_files_are_absent(rel_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert not (repo_root / rel_path).exists(), rel_path


def test_training_config_rejects_duplicate_burst_unlikelihood_objective() -> None:
    objective = list(_canonical_live_stage2_objective())
    objective.insert(1, _duplicate_burst_objective_spec())
    payload = _stage2_training_payload(objective=objective)

    with pytest.raises(ValueError, match=REMOVED_DUPLICATE_BURST_OBJECTIVE):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_config_loader_rejects_temp_yaml_with_duplicate_burst_objective(
    tmp_path: Path,
) -> None:
    payload = _stage2_training_payload(
        objective=[
            _canonical_live_stage2_objective()[0],
            _duplicate_burst_objective_spec(),
            *_canonical_live_stage2_objective()[1:],
        ]
    )
    config_path = tmp_path / "stage2_with_removed_duplicate_burst.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    raw = ConfigLoader.load_yaml_with_extends(str(config_path))
    with pytest.raises(ValueError, match=REMOVED_DUPLICATE_BURST_OBJECTIVE):
        TrainingConfig.from_mapping(raw, PromptOverrides())


@pytest.mark.parametrize("key", sorted(REMOVED_ADJACENT_REPULSION_CONFIG_KEYS))
def test_training_config_rejects_adjacent_repulsion_coord_reg_keys(key: str) -> None:
    objective = [
        *_canonical_live_stage2_objective(),
        {
            "name": "coord_reg",
            "enabled": True,
            "weight": 1.0,
            "application": {"preset": "rollout_self_prefix"},
            "config": {key: 0.0 if key != "adjacent_repulsion_filter_mode" else "same_desc"},
        },
    ]
    payload = _stage2_training_payload(objective=objective)

    with pytest.raises(ValueError, match="coord_reg"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_active_stage2_configs_do_not_declare_duplicate_burst_objective() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    active_config_root = repo_root / "configs" / "stage2_rollout_correction"
    assert active_config_root.exists()

    offenders: list[str] = []
    for path in sorted(active_config_root.rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        if REMOVED_DUPLICATE_BURST_OBJECTIVE in text:
            offenders.append(str(path.relative_to(repo_root)))

    assert offenders == []


def test_active_stage2_configs_do_not_declare_adjacent_repulsion_keys() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    active_config_root = repo_root / "configs" / "stage2_rollout_correction"
    assert active_config_root.exists()

    offenders: list[str] = []
    for path in sorted(active_config_root.rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        for key in sorted(REMOVED_ADJACENT_REPULSION_CONFIG_KEYS):
            if key in text:
                offenders.append(f"{path.relative_to(repo_root)}:{key}")

    assert offenders == []


def test_adjacent_repulsion_training_module_was_removed() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert not (
        repo_root / "src" / "trainers" / "teacher_forcing" / "adjacent_repulsion.py"
    ).exists()
    assert not (repo_root / "tests" / "test_adjacent_repulsion.py").exists()


def test_active_openspec_changes_do_not_reintroduce_adjacent_repulsion_live_support(
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    active_changes_root = repo_root / "openspec" / "changes"

    offenders: list[str] = []
    for change_root in sorted(active_changes_root.iterdir()):
        if not change_root.is_dir():
            continue
        if change_root.name == "archive":
            continue
        if change_root.name in ACTIVE_OPENSPEC_CHANGE_EXCLUSIONS_FOR_REMOVED_MECHANISMS:
            continue

        for path in sorted(change_root.rglob("*.md")):
            lines = path.read_text(encoding="utf-8").splitlines()
            for lineno, line in enumerate(lines, 1):
                normalized = line.lower()

                for key in sorted(REMOVED_ADJACENT_REPULSION_CONFIG_KEYS):
                    if key in line:
                        offenders.append(
                            f"{path.relative_to(repo_root)}:{lineno}:{line.strip()}"
                        )
                for key in sorted(REMOVED_ADJACENT_REPULSION_LIVE_KEYS):
                    if key.lower() in normalized:
                        offenders.append(
                            f"{path.relative_to(repo_root)}:{lineno}:{line.strip()}"
                        )
                for claim in PROHIBITED_ADJACENT_REPULSION_LIVE_SUPPORT_CLAIMS:
                    if claim in normalized:
                        offenders.append(
                            f"{path.relative_to(repo_root)}:{lineno}:{line.strip()}"
                        )

    assert offenders == []


def test_docs_and_catalog_do_not_advertise_duplicate_burst_as_live_training(
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    checked_paths = [
        repo_root / "docs" / "catalog.yaml",
        repo_root / "docs" / "AGENT_INDEX.md",
        repo_root / "docs" / "IMPLEMENTATION_MAP.md",
        repo_root / "docs" / "SYSTEM_OVERVIEW.md",
        repo_root / "docs" / "ARTIFACTS.md",
        repo_root / "docs" / "training" / "README.md",
        repo_root / "docs" / "training" / "STAGE1_OBJECTIVE.md",
        repo_root / "docs" / "training" / "STAGE2_RUNBOOK.md",
        repo_root / "docs" / "training" / "METRICS.md",
    ]

    offenders: list[str] = []
    for path in checked_paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if REMOVED_DUPLICATE_BURST_OBJECTIVE not in line:
                continue
            if any(token in line.lower() for token in ALLOWED_DOC_REMOVAL_CONTEXT):
                continue
            offenders.append(f"{path.relative_to(repo_root)}:{lineno}:{line.strip()}")

    assert offenders == []


def test_docs_and_active_specs_do_not_preserve_stale_duplicate_burst_claims(
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    checked_roots = [
        repo_root / "docs",
        repo_root / "openspec" / "specs",
        repo_root / "openspec" / "changes" / "birth-first-stage2-channel-b",
    ]

    offenders: list[str] = []
    for root in checked_roots:
        for path in sorted(root.rglob("*.md")):
            if "docs/superpowers" in path.as_posix():
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            for lineno, line in enumerate(lines, 1):
                normalized = line.lower()
                for phrase in PROHIBITED_STALE_DUPLICATE_BURST_CLAIMS:
                    if phrase in normalized:
                        if (
                            phrase == "loss-consumed suppression targets"
                            and (
                                "must not create" in normalized
                                or "creates no" in normalized
                            )
                        ):
                            continue
                        offenders.append(
                            f"{path.relative_to(repo_root)}:{lineno}:{line.strip()}"
                        )
                for key in PROHIBITED_STALE_DUPLICATE_METRIC_KEYS:
                    if key.lower() in normalized:
                        offenders.append(
                            f"{path.relative_to(repo_root)}:{lineno}:{line.strip()}"
                        )

                if (
                    root == repo_root / "openspec" / "changes" / "birth-first-stage2-channel-b"
                    and REMOVED_DUPLICATE_BURST_OBJECTIVE in line
                ):
                    context = "\n".join(
                        lines[max(0, lineno - 3) : min(len(lines), lineno + 2)]
                    ).lower()
                    if any(token in context for token in ALLOWED_ACTIVE_OPENSPEC_CONTEXT):
                        continue
                    offenders.append(
                        f"{path.relative_to(repo_root)}:{lineno}:{line.strip()}"
                    )

                if (
                    path.relative_to(repo_root).as_posix()
                    in {
                        "openspec/specs/stage2-ab-training/spec.md",
                        "openspec/specs/channel-b-lightweight-pseudopositive-v1/spec.md",
                    }
                    and "duplicate_iou_threshold" in line
                ):
                    context = "\n".join(
                        lines[max(0, lineno - 8) : min(len(lines), lineno + 2)]
                    ).lower()
                    if "must be rejected" in context or "rejected" in context:
                        continue
                    if "must accept only" in context:
                        offenders.append(
                            f"{path.relative_to(repo_root)}:{lineno}:{line.strip()}"
                        )
                    if (
                        path.relative_to(repo_root).as_posix()
                        == "openspec/specs/channel-b-lightweight-pseudopositive-v1/spec.md"
                        and "configured duplicate-control iou threshold" not in normalized
                    ):
                        offenders.append(
                            f"{path.relative_to(repo_root)}:{lineno}:{line.strip()}"
                        )

    assert offenders == []


def test_metric_writer_does_not_publish_duplicate_burst_live_loss_keys() -> None:
    signature = inspect.signature(build_stage2_core_loss_logs)
    source = inspect.getsource(build_stage2_core_loss_logs)
    repo_root = Path(__file__).resolve().parents[1]
    stage2_trainer_source = (
        repo_root / "src" / "trainers" / "stage2_rollout_correction_impl.py"
    ).read_text(encoding="utf-8")

    assert "duplicate_burst_unlikelihood_module_w" not in signature.parameters
    for key in REMOVED_DUPLICATE_BURST_LIVE_KEYS:
        assert key not in source
        assert key not in stage2_trainer_source
    for key in PROHIBITED_STALE_DUPLICATE_METRIC_KEYS:
        assert key not in source
        assert key not in stage2_trainer_source
    for key in RETAINED_DUPLICATE_DIAGNOSTIC_KEYS:
        assert key in stage2_trainer_source


def test_metric_writers_do_not_publish_adjacent_repulsion_live_loss_keys() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    checked_paths = [
        repo_root / "src" / "trainers" / "metrics" / "coord_losses.py",
        repo_root / "src" / "trainers" / "teacher_forcing" / "objective_atoms.py",
        repo_root / "src" / "trainers" / "rollout_correction" / "objective_runner.py",
        repo_root / "src" / "trainers" / "monitoring" / "loss_gradient_monitor.py",
    ]

    offenders: list[str] = []
    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        for key in sorted(REMOVED_ADJACENT_REPULSION_LIVE_KEYS):
            if key in source:
                offenders.append(f"{path.relative_to(repo_root)}:{key}")

    assert offenders == []


def test_small_object_duplication_study_does_not_emit_stale_duplicate_burst_field() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_path = repo_root / "src" / "analysis" / "small_object_duplication_study.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    emitted_keys: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                emitted_keys.append(key.value)

    assert STALE_DUPLICATE_BURST_ANALYSIS_ARTIFACT_FIELD not in emitted_keys
    assert CANONICAL_DUPLICATE_CONTROL_ANALYSIS_ARTIFACT_FIELD in emitted_keys


def test_live_source_and_tests_do_not_expose_duplicate_burst_ul_targets() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    checked_roots = [repo_root / "src", repo_root / "tests"]
    forbidden = (
        "duplicate_burst_unlikelihood_targets",
        "_build_duplicate_burst_unlikelihood_targets",
        "Stage2DuplicateBurstUnlikelihoodTarget",
        "duplicate-ul positives",
        "duplicate UL is realized",
        "duplicate UL still activates",
        "suppression targets",
    )

    offenders: list[str] = []
    for root in checked_roots:
        for path in sorted(root.rglob("*.py")):
            if path == Path(__file__).resolve():
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            for lineno, line in enumerate(lines, 1):
                normalized = line.lower()
                for phrase in forbidden:
                    if phrase.lower() in normalized:
                        offenders.append(
                            f"{path.relative_to(repo_root)}:{lineno}:{line.strip()}"
                        )

    assert offenders == []
