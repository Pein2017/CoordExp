from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path

import pytest
import torch

import scripts.research.launch_human13_k_trajectory_rp_crossover as launcher
import scripts.research.train_human13_k_trajectory_rp_crossover as runner
from scripts.research.human13_adamw_proposal_preservation import (
    FrozenWitnessBank,
    OwnerWitness,
    ParameterLayout,
    SOURCE_MEMBERSHIPS,
    TRUSTED_OWNER_CLASS,
    WEAKEST_MARGIN_SELECTION,
    WitnessBinding,
    jacobian_sha256,
)
from scripts.research.human13_rp_crossover_runtime import (
    UNIT_ID,
    CellExecutionState,
    ObjectiveBackwardReceipt,
    PrivateCheckpointRef,
)
from scripts.research.human13_training_transaction import (
    TrainingStateTransaction,
    UpdateCounter,
)

from scripts.research.analyze_human13_k_trajectory_rp_crossover import (
    MatrixAnalysisInput,
    MatrixAnalysisReceipt,
    _analyze_matrix_for_test as analyze_matrix,
    analyze_matrix as public_analyze_matrix,
    publish_matrix_analysis_input,
)
from scripts.research.analyze_human13_k_union import _manifest_sha256
from scripts.research.build_human13_k_union_manifest import (
    ArmIdentity,
    GlobalDenominatorIdentity,
    Human13KUnionManifest,
    ImageRecord,
    OwnerRecord,
    PrefixRecord,
    RequestIdentity,
    TrajectoryRecord,
    default_binding,
)
from scripts.research.human13_rp_crossover_matrix_contracts import (
    ARM_IDS,
    CANONICAL_IMAGE_IDS,
    EVALUATION_RPS,
    MATRIX_SEED_GROUPS,
    PROPOSAL_COMPONENTS_BY_ARM,
    QUALIFICATION_SEED_GROUP,
    TRAINING_RPS,
    AcquisitionKey,
    AuditRef,
    CellKey,
    CellReceipt,
    CellSpec,
    MatrixPlan,
    NodeTerminalReceipt,
    SharedEvidenceRef,
    SourceBaselineRef,
    canonical_seeds,
)


SOURCE = "a" * 64


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _content_sha256(value: dict[str, object]) -> str:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_trajectory(image_id: int) -> TrajectoryRecord:
    tokens = (101, 102, 999)
    return TrajectoryRecord(
        trajectory_id=f"source:{image_id}",
        request=RequestIdentity(
            backend="hf",
            backend_version="test-hf",
            mode="source_greedy",
            n=1,
            seed=None,
            physical_batch_index=0,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_new_tokens=512,
        ),
        raw_token_ids=tokens,
        terminal_token_index=2,
        stop_reason="im_end",
        parser_status="complete",
        rows=(),
        prefix=PrefixRecord(
            raw_token_ids=tokens[:-1],
            clean_token_ids=tokens[:-1],
            removed_row_ids=(),
        ),
        retained_row_ids=(),
        duplicate_row_ids=(),
        matched_row_ids=(),
        replay_token_mask=(False, False, False),
        duplicate_target_mask=(False, False, False),
    )


def _manifest() -> Human13KUnionManifest:
    images: list[ImageRecord] = []
    for image_id in CANONICAL_IMAGE_IDS:
        owners = (
            OwnerRecord(
                f"g:{image_id}",
                "person",
                (0.0, 0.0, 10.0, 10.0),
                0,
                "G",
                (),
                (),
            ),
            OwnerRecord(
                f"h:{image_id}",
                "person",
                (20.0, 0.0, 30.0, 10.0),
                1,
                "H",
                (),
                (),
            ),
            OwnerRecord(
                f"m:{image_id}",
                "person",
                (40.0, 0.0, 50.0, 10.0),
                2,
                "M",
                (),
                (),
            ),
        )
        images.append(
            ImageRecord(
                image_id=image_id,
                panel_row_sha256=None,
                image_sha256=None,
                owners=owners,
                trajectories=(_source_trajectory(image_id),),
                duplicate_events=(),
                selected_rows=(),
                g_owner_ids=(f"g:{image_id}",),
                h_owner_ids=(f"h:{image_id}",),
                m_owner_ids=(f"m:{image_id}",),
                replay_row_ids=(),
                target_row_ids=(),
                candidate_row_ids=(),
            )
        )
    return Human13KUnionManifest(
        schema_version="human13_k_union_manifest.v1",
        binding=default_binding(),
        images=tuple(images),
        arms=tuple(ArmIdentity(arm, "H", True) for arm in ARM_IDS),
        denominators=GlobalDenominatorIdentity(13, 0, 0, 0, 0, 0, 0),
        full_panel=False,
    )


ADAMW_CONFIG = _digest("frozen-adamw-config")


def _shared(manifest_sha256: str, rp: float, seed_group: str) -> SharedEvidenceRef:
    tag = f"{rp}:{seed_group}"
    return SharedEvidenceRef(
        source_sha256=SOURCE,
        manifest_sha256=manifest_sha256,
        acquisition_path=f"/sealed/acquisition/{tag}.json",
        acquisition_sha256=_digest(f"acquisition:{tag}"),
        trajectory_credit_acquisition_sha256=_digest(f"credit-acq:{tag}"),
        credit_ledger_sha256=_digest(f"credit:{tag}"),
        compiler_ledger_sha256=_digest(f"compiler:{tag}"),
        policy_contract_sha256=_digest(f"policy:{rp}"),
        native_receipts_sha256=_digest(f"native-receipts:{tag}"),
        training_rp=rp,
        seed_group_id=seed_group,
        seeds=canonical_seeds(seed_group),
    )


def _objective_hashes(arm_id: str, group_tag: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (component, _digest(f"objective:{component}:{group_tag}"))
        for component in PROPOSAL_COMPONENTS_BY_ARM[arm_id]
    )


def _proposal_delta(arm_id: str, group_tag: str) -> str:
    """Arm C projects the exact arm-B base proposal; arm A measures its own."""

    stage = "a" if arm_id == "A" else "bc"
    return _digest(f"proposal-delta:{group_tag}:{stage}")


def _predictions(
    image_id: int,
    *,
    include_g: bool,
    include_h: bool,
    include_m: bool,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    if include_g:
        result.append(
            {"generated_order": 0, "description": "person", "bbox": [0, 0, 10, 10]}
        )
    if include_h:
        result.append(
            {"generated_order": 1, "description": "person", "bbox": [20, 0, 30, 10]}
        )
    if include_m:
        result.append(
            {"generated_order": 2, "description": "person", "bbox": [40, 0, 50, 10]}
        )
    if result:
        duplicate = dict(result[0])
        duplicate["generated_order"] = 3
        result.append(duplicate)
    result.extend(
        (
            {"generated_order": 4, "description": "person", "bbox": None},
            {
                "generated_order": 5,
                "description": "bicycle",
                "bbox": [70, 0, 80, 10],
            },
        )
    )
    # Deliberately reverse storage order: chronological duplicate projection
    # must still follow generated_order.
    return list(reversed(result))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _audit_rows(
    manifest: Human13KUnionManifest,
    *,
    rp: float,
    checkpoint_sha256: str,
    generation_sha256: str,
    policy_sha256: str,
    config_sha256: str,
    arm_id: str,
    seed_group: str,
    training_rp: float,
    mixed_robust_seeds: bool,
    source: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    passing = seed_group in {"matrix_a", "matrix_b"}
    if mixed_robust_seeds:
        passing = seed_group in (
            {"matrix_a", "matrix_b"} if rp == 1.0 else {"matrix_b", "matrix_c"}
        )
    for index, image in enumerate(manifest.images):
        if source:
            include_g = True
            include_h = False
        elif arm_id == "A":
            include_g = True
            include_h = False
        elif arm_id == "B":
            include_g = True
            include_h = True
        else:
            include_g = passing
            include_h = True
        include_m = index == 0 and (
            source
            and rp == 1.0
            or arm_id == "B"
            and rp == 1.10
            or arm_id == "C"
            and passing
            and rp == 1.0
        )
        rows.append(
            {
                "image_id": image.image_id,
                "decode_mode": "original_prompt_clean_greedy",
                "repetition_penalty": rp,
                "generation_policy_receipt_sha256": generation_sha256,
                "predictions": _predictions(
                    image.image_id,
                    include_g=include_g,
                    include_h=include_h,
                    include_m=include_m,
                ),
                "generated_token_ids": [11, 12, 13, 14, 999],
                "malformed_row_count": 1,
                "stop_reason": "im_end",
                "provenance": {
                    "manifest_sha256": _manifest_sha256(manifest),
                    "source_checkpoint_sha256": SOURCE,
                    "checkpoint_payload_sha256": checkpoint_sha256,
                    "policy_contract_sha256": policy_sha256,
                    "resolved_config_sha256": config_sha256,
                    "training_rp": training_rp,
                    "seed_group_id": seed_group,
                    "arm_id": arm_id,
                },
            }
        )
    return rows


def _fixture(
    tmp_path: Path, *, mixed_robust_seeds: bool = False
) -> tuple[Human13KUnionManifest, MatrixAnalysisInput]:
    manifest = _manifest()
    manifest_sha = _manifest_sha256(manifest)
    source_paths: list[tuple[float, str]] = []
    baselines: list[SourceBaselineRef] = []
    for rp in EVALUATION_RPS:
        path = tmp_path / f"source-rp{rp}.jsonl"
        output_sha = _write_jsonl(
            path,
            _audit_rows(
                manifest,
                rp=rp,
                checkpoint_sha256=SOURCE,
                generation_sha256=_digest(f"source-generation:{rp}"),
                policy_sha256=_digest(f"source-policy:{rp}"),
                config_sha256=_digest(f"source-config:{rp}"),
                arm_id="source",
                seed_group="source",
                training_rp=rp,
                mixed_robust_seeds=mixed_robust_seeds,
                source=True,
            ),
        )
        source_paths.append((rp, str(path)))
        baselines.append(
            SourceBaselineRef(rp, output_sha, output_sha, SOURCE, CANONICAL_IMAGE_IDS)
        )

    acquisitions: list[AcquisitionKey] = []
    cells: list[CellSpec] = []
    receipts: list[CellReceipt] = []
    for training_rp in (1.0, 1.10):
        for seed_group in MATRIX_SEED_GROUPS:
            acquisition = AcquisitionKey(training_rp, seed_group, "matrix")
            acquisitions.append(acquisition)
            shared = _shared(manifest_sha, training_rp, seed_group)
            for arm_id in ARM_IDS:
                cell_tag = f"{training_rp}:{seed_group}:{arm_id}"
                cell_key = CellKey(acquisition, arm_id)
                components = {
                    "A": ("trajectory",),
                    "B": ("trajectory", "compiler"),
                    "C": ("trajectory", "compiler", "preservation"),
                }[arm_id]
                config_sha = _digest(f"config:{training_rp}:{arm_id}")
                group_tag = f"{training_rp}:{seed_group}"
                cell = CellSpec(
                    cell_key=cell_key,
                    shared_evidence=shared,
                    leaf_config_sha256=config_sha,
                    source_checkpoint_sha256=SOURCE,
                    expected_objective_components=components,
                    objective_component_hashes=_objective_hashes(arm_id, group_tag),
                    adamw_config_sha256=ADAMW_CONFIG,
                    fresh_optimizer_identity_sha256=_digest(f"optimizer:{cell_tag}"),
                    evaluation_rps=EVALUATION_RPS,
                    output_root=f"/private/{cell_tag}",
                )
                cells.append(cell)
                checkpoint_sha = _digest(f"checkpoint:{cell_tag}")
                audits: list[AuditRef] = []
                for eval_rp in EVALUATION_RPS:
                    generation_sha = _digest(f"generation:{cell_tag}:{eval_rp}")
                    output_path = tmp_path / f"audit-{cell_tag}-{eval_rp}.jsonl"
                    output_sha = _write_jsonl(
                        output_path,
                        _audit_rows(
                            manifest,
                            rp=eval_rp,
                            checkpoint_sha256=checkpoint_sha,
                            generation_sha256=generation_sha,
                            policy_sha256=shared.policy_contract_sha256,
                            config_sha256=config_sha,
                            arm_id=arm_id,
                            seed_group=seed_group,
                            training_rp=training_rp,
                            mixed_robust_seeds=mixed_robust_seeds,
                        ),
                    )
                    audits.append(
                        AuditRef(
                            eval_rp,
                            checkpoint_sha,
                            str(output_path),
                            output_sha,
                            13,
                            CANONICAL_IMAGE_IDS,
                            generation_sha,
                        )
                    )
                transaction = _digest("fresh-source-state")
                receipts.append(
                    CellReceipt(
                        cell_key=cell_key,
                        shared_evidence=shared,
                        objective_components=components,
                        objective_component_hashes=cell.objective_component_hashes,
                        adamw_config_sha256=ADAMW_CONFIG,
                        fresh_optimizer_identity_sha256=(
                            cell.fresh_optimizer_identity_sha256
                        ),
                        transaction_id=_digest(f"transaction-id:{cell_tag}")[:32],
                        before_transaction_digest=transaction,
                        after_transaction_digest=transaction,
                        status="succeeded",
                        audits=tuple(audits),
                        adamw_proposal_sha256=_digest(f"proposal:{cell_tag}"),
                        proposal_delta_sha256=_proposal_delta(arm_id, group_tag),
                        projection_receipt_sha256=(
                            _digest(f"projection:{cell_tag}") if arm_id == "C" else None
                        ),
                        apply_receipt_sha256=_digest(f"apply:{cell_tag}"),
                    )
                )
    acquisitions_tuple = tuple(acquisitions)
    cells_tuple = tuple(cells)
    plan = MatrixPlan(
        acquisitions=acquisitions_tuple,
        cells=cells_tuple,
        source_baselines=tuple(baselines),
        dependency_edges=tuple(
            (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
            for cell in cells_tuple
        ),
        concurrency_cap=8,
        dry_run_counters={
            "model_loads": 0,
            "gpu_allocations": 0,
            "subprocess_launches": 0,
            "output_roots_created": 0,
        },
    )
    return manifest, MatrixAnalysisInput(plan, tuple(receipts), tuple(source_paths))


def _surface(
    result: MatrixAnalysisReceipt,
    *,
    training_rp: float,
    seed_group: str,
    arm_id: str,
    evaluation_rp: float,
):
    return next(
        item
        for item in result.surfaces
        if (
            item.training_rp,
            item.seed_group_id,
            item.arm_id,
            item.evaluation_rp,
        )
        == (training_rp, seed_group, arm_id, evaluation_rp)
    )


def test_complete_matrix_reports_separate_owner_and_burden_channels(
    tmp_path: Path,
) -> None:
    manifest, inputs = _fixture(tmp_path)

    result = analyze_matrix(manifest, inputs)

    assert len(result.surfaces) == 36
    surface = _surface(
        result,
        training_rp=1.0,
        seed_group="matrix_a",
        arm_id="B",
        evaluation_rp=1.10,
    )
    assert len(surface.trusted_gain_owner_ids) == 13
    assert surface.baseline_loss_owner_ids == ()
    assert len(surface.historical_g_owner_ids) == 13
    assert len(surface.historical_h_owner_ids) == 13
    assert surface.historical_m_owner_ids == (f"m:{CANONICAL_IMAGE_IDS[0]}",)
    assert surface.incidental_m_recovery_owner_ids == (f"m:{CANONICAL_IMAGE_IDS[0]}",)
    assert surface.protected_but_undefendable_m_owner_ids == ()
    assert surface.duplicate_rows == 13
    assert surface.malformed_rows == 13
    assert surface.invalid_rows == 13
    assert surface.unmatched_rows == 13
    assert surface.prediction_rows == 79
    assert surface.generated_tokens == 65
    assert surface.stop_reason_counts == (("im_end", 13),)
    assert len(surface.legacy_union_at_k_diagnostic_owner_ids) == 26
    assert surface.union_at_k_is_success_evidence is False

    source_m = _surface(
        result,
        training_rp=1.0,
        seed_group="matrix_a",
        arm_id="A",
        evaluation_rp=1.0,
    )
    assert source_m.protected_but_undefendable_m_owner_ids == (
        f"m:{CANONICAL_IMAGE_IDS[0]}",
    )


def test_paired_changes_and_success_use_only_same_rp_seed_groups(
    tmp_path: Path,
) -> None:
    manifest, inputs = _fixture(tmp_path)
    result = analyze_matrix(manifest, inputs)

    a_to_b = next(
        item
        for item in result.paired_changes
        if (
            item.training_rp,
            item.seed_group_id,
            item.evaluation_rp,
            item.from_arm,
            item.to_arm,
        )
        == (1.0, "matrix_a", 1.0, "A", "B")
    )
    assert len(a_to_b.added_trusted_gain_owner_ids) == 13
    assert a_to_b.added_baseline_loss_owner_ids == ()

    success = {item.training_rp: item for item in result.success}
    assert success[1.0].contract_local_passing_seed_groups == (
        "matrix_a",
        "matrix_b",
    )
    assert success[1.0].rp_robust_passing_seed_groups == (
        "matrix_a",
        "matrix_b",
    )
    assert success[1.0].contract_local_success is True
    assert success[1.0].rp_robust_success is True
    assert result.bi_policy_replication is True


def test_rp_robust_success_never_mixes_passing_seeds_across_surfaces(
    tmp_path: Path,
) -> None:
    manifest, inputs = _fixture(tmp_path, mixed_robust_seeds=True)

    result = analyze_matrix(manifest, inputs)

    success = {item.training_rp: item for item in result.success}
    assert success[1.0].contract_local_success is True
    assert success[1.0].rp_robust_passing_seed_groups == ("matrix_b",)
    assert success[1.0].rp_robust_success is False
    assert success[1.10].contract_local_passing_seed_groups == (
        "matrix_b",
        "matrix_c",
    )
    assert success[1.10].rp_robust_passing_seed_groups == ("matrix_b",)


def test_direct_dict_and_file_inputs_share_one_admission_path(tmp_path: Path) -> None:
    manifest, inputs = _fixture(tmp_path)
    direct = analyze_matrix(manifest, inputs)
    mapped = analyze_matrix(manifest, inputs.to_dict())
    path = tmp_path / "analysis-input.json"
    path.write_text(json.dumps(inputs.to_dict()), encoding="utf-8")
    loaded = analyze_matrix(manifest, path)

    assert direct.content_sha256 == mapped.content_sha256 == loaded.content_sha256
    assert MatrixAnalysisReceipt.from_dict(direct.to_dict()) == direct


@pytest.mark.parametrize(
    "mutation, message",
    (
        ("missing", "exactly eighteen"),
        ("duplicate", "unique cell"),
        ("failed", "failed scientific cell"),
        ("shared", "shared evidence"),
        ("transaction", "independent transaction identity"),
        ("proposal", "independent proposal"),
        ("base_proposal", "exact admitted arm-B proposal"),
        ("objective", "objective component"),
        ("optimizer", "optimizer identity"),
        ("qualification", "qualification"),
    ),
)
def test_aggregate_admission_rejects_noncanonical_receipts(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _manifest_unused, inputs = _fixture(tmp_path)
    receipts = list(inputs.cell_receipts)
    if mutation == "missing":
        receipts.pop()
    elif mutation == "duplicate":
        receipts[-1] = receipts[0]
    elif mutation == "failed":
        receipts[0] = replace(
            receipts[0],
            status="failed",
            failure_reason="scientific failure",
            audits=(),
            adamw_proposal_sha256=None,
            proposal_delta_sha256=None,
            apply_receipt_sha256=None,
            objective_component_hashes=(),
        )
    elif mutation == "shared":
        receipts[0] = replace(
            receipts[0],
            shared_evidence=replace(
                receipts[0].shared_evidence,
                credit_ledger_sha256=_digest("mixed-credit"),
            ),
        )
    elif mutation == "transaction":
        receipts[1] = replace(receipts[1], transaction_id=receipts[0].transaction_id)
    elif mutation == "proposal":
        receipts[1] = replace(
            receipts[1], adamw_proposal_sha256=receipts[0].adamw_proposal_sha256
        )
    elif mutation == "base_proposal":
        receipts[2] = replace(
            receipts[2], proposal_delta_sha256=_digest("another-base-proposal")
        )
    elif mutation == "objective":
        receipts[0] = replace(
            receipts[0],
            objective_component_hashes=(("trajectory", _digest("forged")),),
        )
    elif mutation == "optimizer":
        receipts[1] = replace(
            receipts[1],
            fresh_optimizer_identity_sha256=(
                receipts[0].fresh_optimizer_identity_sha256
            ),
        )
    else:
        qualification = AcquisitionKey(1.0, QUALIFICATION_SEED_GROUP, "qualification")
        receipts[0] = replace(
            receipts[0], cell_key=CellKey(qualification, receipts[0].cell_key.arm_id)
        )
    with pytest.raises(ValueError, match=message):
        MatrixAnalysisInput(inputs.plan, tuple(receipts), inputs.source_output_paths)


@pytest.mark.parametrize(
    "mutation, message",
    (
        ("hash", "SHA-256"),
        ("rp", "generation RP"),
        ("images", "thirteen image IDs"),
        ("manifest", "manifest lineage"),
        ("policy", "policy lineage"),
        ("config", "config lineage"),
    ),
)
def test_audit_loader_rejects_mixed_or_forged_output_artifacts(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest, inputs = _fixture(tmp_path)
    receipts = list(inputs.cell_receipts)
    receipt = receipts[0]
    audit = receipt.audits[0]
    path = Path(audit.output_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if mutation == "hash":
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        if mutation == "rp":
            rows[0]["repetition_penalty"] = 1.10
        elif mutation == "images":
            rows.pop()
        elif mutation == "manifest":
            rows[0]["provenance"]["manifest_sha256"] = "f" * 64
        elif mutation == "policy":
            rows[0]["provenance"]["policy_contract_sha256"] = "f" * 64
        else:
            rows[0]["provenance"]["resolved_config_sha256"] = "f" * 64
        new_sha = _write_jsonl(path, rows)
        changed_audit = replace(audit, output_sha256=new_sha)
        receipts[0] = replace(receipt, audits=(changed_audit, receipt.audits[1]))
    forged = MatrixAnalysisInput(
        inputs.plan, tuple(receipts), inputs.source_output_paths
    )
    with pytest.raises(ValueError, match=message):
        analyze_matrix(manifest, forged)


def test_output_receipt_is_immutable_and_content_addressed(tmp_path: Path) -> None:
    manifest, inputs = _fixture(tmp_path)
    result = analyze_matrix(manifest, inputs)
    payload = result.to_dict()
    payload["surfaces"][0]["trusted_gain_owner_ids"].append("forged")

    with pytest.raises(ValueError, match="content SHA-256"):
        MatrixAnalysisReceipt.from_dict(payload)


def test_output_receipt_rejects_rehashed_duplicate_surface_identity(
    tmp_path: Path,
) -> None:
    manifest, inputs = _fixture(tmp_path)
    payload = analyze_matrix(manifest, inputs).to_dict()
    payload["surfaces"][1]["training_rp"] = payload["surfaces"][0]["training_rp"]
    payload["surfaces"][1]["seed_group_id"] = payload["surfaces"][0]["seed_group_id"]
    payload["surfaces"][1]["arm_id"] = payload["surfaces"][0]["arm_id"]
    payload["surfaces"][1]["evaluation_rp"] = payload["surfaces"][0]["evaluation_rp"]
    preimage = {key: value for key, value in payload.items() if key != "content_sha256"}
    payload["content_sha256"] = _content_sha256(preimage)

    with pytest.raises(ValueError, match="complete surface identities"):
        MatrixAnalysisReceipt.from_dict(payload)


def test_public_analyzer_requires_the_exact_frozen_manifest(tmp_path: Path) -> None:
    manifest, inputs = _fixture(tmp_path)

    with pytest.raises(ValueError, match="frozen manifest SHA-256"):
        public_analyze_matrix(manifest, inputs)


def test_analyzer_rejects_a_noncanonical_matcher_identity(tmp_path: Path) -> None:
    manifest, inputs = _fixture(tmp_path)
    forged = replace(
        manifest,
        binding=replace(
            manifest.binding,
            matcher=replace(manifest.binding.matcher, algorithm="greedy"),
        ),
    )

    with pytest.raises(ValueError, match="canonical matcher"):
        analyze_matrix(forged, inputs)


# ---------------------------------------------------------------------------
# One launcher-to-analyzer admission path
# ---------------------------------------------------------------------------


def _grad_for(training_rp: float, seed_group: str, arm_id: str) -> torch.Tensor:
    """One deterministic gradient per acquisition; arms B and C share it."""

    index = TRAINING_RPS.index(training_rp) * 3 + MATRIX_SEED_GROUPS.index(seed_group)
    offset = 0.0 if arm_id == "A" else 0.125
    return torch.tensor((0.25 + 0.01 * index + offset, -0.5 - 0.01 * index))


class _E2ECellServices:
    """CPU-only stand-in for the live model/loss/checkpoint/HF audit adapter."""

    def __init__(
        self, spec: CellSpec, manifest: Human13KUnionManifest, root: Path
    ) -> None:
        self.spec = spec
        self.manifest = manifest
        self.root = root
        acquisition = spec.cell_key.acquisition_key
        self.training_rp = acquisition.training_rp
        self.seed_group = acquisition.seed_group_id
        self.arm_id = spec.cell_key.arm_id
        self.tag = f"{self.training_rp}:{self.seed_group}:{self.arm_id}"
        self.checkpoint_sha256 = _digest(f"private-checkpoint:{self.tag}")

    def open_cell(self, spec: CellSpec) -> CellExecutionState:
        parameter = torch.nn.Parameter(torch.tensor((1.0, -1.0)))
        named = (("adapter.weight", parameter),)
        optimizer = torch.optim.AdamW(
            (parameter,), lr=3e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        counter = UpdateCounter()
        return CellExecutionState(
            named_trainable_parameters=named,
            optimizer=optimizer,
            scheduler=None,
            update_counter=counter,
            transaction=TrainingStateTransaction(
                named,
                optimizer=optimizer,
                scheduler=None,
                update_counter=counter,
                runtime=None,
                capture_cuda=False,
            ),
            world_size=1,
            source_checkpoint_sha256=spec.source_checkpoint_sha256,
            shared_evidence_sha256=spec.shared_evidence.content_sha256,
            adamw_config_sha256=spec.adamw_config_sha256,
            optimizer_identity_sha256=spec.fresh_optimizer_identity_sha256,
            fresh_source=True,
        )

    def backward_objective(
        self, state: CellExecutionState, spec: CellSpec
    ) -> ObjectiveBackwardReceipt:
        state.named_trainable_parameters[0][1].grad = _grad_for(
            self.training_rp, self.seed_group, self.arm_id
        )
        return ObjectiveBackwardReceipt(
            proposal_components=tuple(
                component for component, _ in spec.objective_component_hashes
            ),
            component_hashes=spec.objective_component_hashes,
            objective_ledger_sha256=_digest(f"objective-ledger:{self.tag}"),
            shared_evidence_sha256=spec.shared_evidence.content_sha256,
            trajectory_credit_acquisition_sha256=(
                spec.shared_evidence.trajectory_credit_acquisition_sha256
            ),
            compiler_ledger_sha256=spec.shared_evidence.compiler_ledger_sha256,
            backward_count=1,
            optimizer_step_count=0,
        )

    def witness_bank(
        self, state: CellExecutionState, spec: CellSpec
    ) -> FrozenWitnessBank:
        layout = ParameterLayout.from_named_parameters(state.named_trainable_parameters)
        row = torch.ones(layout.total_numel, dtype=torch.float64)
        witness = OwnerWitness(
            image_id="image_00",
            owner_id="owner_00",
            source_membership=SOURCE_MEMBERSHIPS[0],
            owner_class=TRUSTED_OWNER_CLASS,
            token_id=123,
            margin_selection=WEAKEST_MARGIN_SELECTION,
            margin_value=0.5,
            detached=True,
            jacobian_sha256=jacobian_sha256(row),
        )
        return FrozenWitnessBank.from_witnesses(
            (witness,),
            jacobians={witness.canonical_key: row},
            binding=WitnessBinding(
                unit_id=UNIT_ID,
                source_checkpoint_sha256=spec.source_checkpoint_sha256,
                manifest_sha256=spec.shared_evidence.manifest_sha256,
                frozen_before_acquisition=True,
            ),
            layout=layout,
        )

    def realized_margin_probe(self, state, spec, bank):
        return lambda: {
            witness.canonical_key: witness.margin_value + 0.1
            for witness in bank.constraints
        }

    def write_private_checkpoint(
        self, state: CellExecutionState, spec: CellSpec
    ) -> PrivateCheckpointRef:
        return PrivateCheckpointRef(
            path=str(self.root / self.tag / "proposal"),
            checkpoint_sha256=self.checkpoint_sha256,
            private=True,
        )

    def audit_checkpoint(
        self,
        state: CellExecutionState,
        checkpoint: PrivateCheckpointRef,
        repetition_penalty: float,
    ) -> AuditRef:
        generation_sha = _digest(f"generation:{self.tag}:{repetition_penalty}")
        output_path = self.root / f"audit-{self.tag}-{repetition_penalty}.jsonl"
        output_sha = _write_jsonl(
            output_path,
            _audit_rows(
                self.manifest,
                rp=repetition_penalty,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
                generation_sha256=generation_sha,
                policy_sha256=self.spec.shared_evidence.policy_contract_sha256,
                config_sha256=self.spec.leaf_config_sha256,
                arm_id=self.arm_id,
                seed_group=self.seed_group,
                training_rp=self.training_rp,
                mixed_robust_seeds=False,
            ),
        )
        return AuditRef(
            evaluation_rp=repetition_penalty,
            evaluated_checkpoint_sha256=checkpoint.checkpoint_sha256,
            output_path=str(output_path),
            output_sha256=output_sha,
            row_count=13,
            image_ids=CANONICAL_IMAGE_IDS,
            generation_policy_receipt_sha256=generation_sha,
        )

    def cleanup_private_checkpoint(self, checkpoint: PrivateCheckpointRef) -> None:
        return None


class _E2ENodeRuntime:
    """Injected node owner: one acquisition, three independent cells."""

    def __init__(
        self, node: Mapping[str, object], manifest: Human13KUnionManifest, root: Path
    ) -> None:
        self.node = node
        self.manifest = manifest
        self.root = root
        self.terminals: list[tuple[str, Mapping[str, object]]] = []

    def acquire_cell_specs(self, node: Mapping[str, object]) -> tuple[CellSpec, ...]:
        acquisition = AcquisitionKey.from_dict(node["acquisition_key"])
        shared = _shared(
            _manifest_sha256(self.manifest),
            acquisition.training_rp,
            acquisition.seed_group_id,
        )
        group_tag = f"{acquisition.training_rp}:{acquisition.seed_group_id}"
        specs = []
        for cell in node["cells"]:
            arm_id = cell["cell_key"]["arm_id"]
            specs.append(
                CellSpec(
                    cell_key=CellKey(acquisition, arm_id),
                    shared_evidence=shared,
                    leaf_config_sha256=_digest(
                        f"config:{acquisition.training_rp}:{arm_id}"
                    ),
                    source_checkpoint_sha256=SOURCE,
                    expected_objective_components=tuple(cell["objective_components"]),
                    objective_component_hashes=_objective_hashes(arm_id, group_tag),
                    adamw_config_sha256=cell["adamw_config_sha256"],
                    fresh_optimizer_identity_sha256=cell[
                        "fresh_optimizer_identity_sha256"
                    ],
                    evaluation_rps=EVALUATION_RPS,
                    output_root=cell["output_root"],
                )
            )
        return tuple(specs)

    def services_for_cell(self, spec: CellSpec) -> _E2ECellServices:
        return _E2ECellServices(spec, self.manifest, self.root)

    def write_cell_receipt(self, receipt: CellReceipt) -> None:
        return None

    def write_node_terminal_receipt(
        self, path: str, payload: Mapping[str, object]
    ) -> None:
        self.terminals.append((path, payload))


def _source_baseline_fixtures(
    manifest: Human13KUnionManifest, tmp_path: Path
) -> tuple[tuple[SourceBaselineRef, ...], tuple[tuple[float, str], ...]]:
    baselines: list[SourceBaselineRef] = []
    paths: list[tuple[float, str]] = []
    for rp in EVALUATION_RPS:
        path = tmp_path / f"source-rp{rp}.jsonl"
        output_sha = _write_jsonl(
            path,
            _audit_rows(
                manifest,
                rp=rp,
                checkpoint_sha256=SOURCE,
                generation_sha256=_digest(f"source-generation:{rp}"),
                policy_sha256=_digest(f"source-policy:{rp}"),
                config_sha256=_digest(f"source-config:{rp}"),
                arm_id="source",
                seed_group="source",
                training_rp=rp,
                mixed_robust_seeds=False,
                source=True,
            ),
        )
        baselines.append(
            SourceBaselineRef(rp, output_sha, output_sha, SOURCE, CANONICAL_IMAGE_IDS)
        )
        paths.append((rp, str(path)))
    return tuple(baselines), tuple(paths)


def _run_matrix_nodes(tmp_path: Path, manifest: Human13KUnionManifest) -> list[dict]:
    dag_plan = launcher.build_dag_plan(
        launcher.load_leaf_configs(),
        run_id="e2e",
        output_root=tmp_path / "artifacts",
    )
    dag_path = tmp_path / "dag-plan.json"
    dag_path.write_text(json.dumps(dag_plan), encoding="utf-8")
    matrix_node_ids = tuple(
        item["node_id"]
        for item in dag_plan["acquisitions"]
        if item["phase"] == "matrix"
    )
    launch_plan = launcher.plan_launches(
        dag_path, gpu_ids=tuple(range(6)), node_ids=matrix_node_ids
    )
    terminals: list[dict] = []
    for job in launch_plan["jobs"]:
        command = job["command"]
        runner_index = command.index(str(launcher.DEFAULT_RUNNER_ENTRY.resolve()))
        argv = command[runner_index + 1 :]
        node = next(
            item
            for item in dag_plan["acquisitions"]
            if item["node_id"] == job["node_id"]
        )
        runtime = _E2ENodeRuntime(node, manifest, tmp_path / job["node_id"])
        stream = io.StringIO()
        assert (
            runner.run_cli(
                argv,
                node_runtime_factory=runner.declare_node_runtime_factory(
                    lambda _node, runtime=runtime: runtime
                ),
                stdout=stream,
            )
            == 0
        )
        terminals.append(json.loads(stream.getvalue()))
    return terminals


def test_launcher_dag_through_node_runner_and_publisher_is_admitted(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    terminals = _run_matrix_nodes(tmp_path, manifest)
    baselines, source_paths = _source_baseline_fixtures(manifest, tmp_path)
    published_path = tmp_path / "analysis-input.json"

    published = publish_matrix_analysis_input(
        terminals,
        source_baselines=baselines,
        source_output_paths=source_paths,
        output_path=published_path,
    )
    reloaded = MatrixAnalysisInput.from_dict(
        json.loads(published_path.read_text(encoding="utf-8"))
    )
    result = analyze_matrix(manifest, reloaded)

    assert len(terminals) == 6
    assert published.content_sha256 == reloaded.content_sha256
    assert len(published.plan.cells) == 18
    assert len(result.surfaces) == 36
    assert result.qualification_outcomes_included is False
    deltas = {
        (
            receipt.cell_key.acquisition_key.training_rp,
            receipt.cell_key.acquisition_key.seed_group_id,
            receipt.cell_key.arm_id,
        ): receipt.proposal_delta_sha256
        for receipt in published.cell_receipts
    }
    # Arm C projects exactly the arm-B base proposal of its own acquisition.
    for training_rp in TRAINING_RPS:
        for seed_group in MATRIX_SEED_GROUPS:
            assert (
                deltas[(training_rp, seed_group, "C")]
                == deltas[(training_rp, seed_group, "B")]
            )


def test_publisher_rejects_a_qualification_node_terminal(tmp_path: Path) -> None:
    manifest = _manifest()
    terminals = _run_matrix_nodes(tmp_path, manifest)
    baselines, source_paths = _source_baseline_fixtures(manifest, tmp_path)
    qualification = NodeTerminalReceipt.from_dict(terminals[0])
    forged = replace(
        qualification,
        node_id="rp100:qualification",
        phase="qualification",
        acquisition_key=AcquisitionKey(1.0, QUALIFICATION_SEED_GROUP, "qualification"),
        cell_specs=(),
        cell_receipts=(),
        status="failed",
        failure_reason="not a matrix node",
    )

    with pytest.raises(ValueError, match="qualification"):
        publish_matrix_analysis_input(
            [*terminals, forged],
            source_baselines=baselines,
            source_output_paths=source_paths,
        )


def test_publisher_rejects_a_missing_node_terminal(tmp_path: Path) -> None:
    manifest = _manifest()
    terminals = _run_matrix_nodes(tmp_path, manifest)
    baselines, source_paths = _source_baseline_fixtures(manifest, tmp_path)

    with pytest.raises(ValueError, match="exactly six"):
        publish_matrix_analysis_input(
            terminals[:-1],
            source_baselines=baselines,
            source_output_paths=source_paths,
        )
