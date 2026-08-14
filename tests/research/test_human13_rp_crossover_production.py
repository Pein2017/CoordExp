from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import pytest

from scripts.research import human13_rp_crossover_production as production
from scripts.research import launch_human13_k_trajectory_rp_crossover as launcher
from scripts.research import train_human13_k_trajectory_rp_crossover as runner
from scripts.research.human13_rp_crossover_matrix_contracts import (
    CANONICAL_IMAGE_IDS,
    AcquisitionKey,
    AuditRef,
    CellKey,
    CellReceipt,
    CellSpec,
    SharedEvidenceRef,
    SourceBaselineRef,
)
from scripts.research.human13_rp_crossover_runtime import CellRuntimeServices


REPO_ROOT = Path(__file__).resolve().parents[2]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _qualification_node(tmp_path: Path, *, rp: float = 1.0) -> dict[str, Any]:
    plan = launcher.build_dag_plan(
        launcher.load_leaf_configs(),
        run_id=f"production-rp-{rp}",
        output_root=tmp_path / "artifacts",
    )
    node_id = "rp100:qualification" if rp == 1.0 else "rp110:qualification"
    return next(item for item in plan["acquisitions"] if item["node_id"] == node_id)


def _qualification_spec(node: Mapping[str, Any]) -> CellSpec:
    acquisition = AcquisitionKey.from_dict(node["acquisition_key"])
    planned = node["cells"][0]
    trajectory = _digest(f"trajectory:{acquisition.training_rp}")
    compiler = _digest(f"compiler:{acquisition.training_rp}")
    source = production.SOURCE_CHECKPOINT_PAYLOAD_SHA256
    shared = SharedEvidenceRef(
        source_sha256=source,
        manifest_sha256=production.MANIFEST_SHA256,
        acquisition_path=str(Path(planned["output_root"]).parent / "acquisition"),
        acquisition_sha256=_digest(f"acquisition:{acquisition.training_rp}"),
        trajectory_credit_acquisition_sha256=_digest(
            f"trajectory-acquisition:{acquisition.training_rp}"
        ),
        credit_ledger_sha256=trajectory,
        compiler_ledger_sha256=compiler,
        policy_contract_sha256=_digest(f"policy:{acquisition.training_rp}"),
        native_receipts_sha256=_digest(f"native:{acquisition.training_rp}"),
        training_rp=acquisition.training_rp,
        seed_group_id="qualification",
        seeds=tuple(range(30001, 30017)),
    )
    return CellSpec(
        cell_key=CellKey(acquisition, "C"),
        shared_evidence=shared,
        leaf_config_sha256=production.C_LEAF_SHA256_BY_RP[acquisition.training_rp],
        source_checkpoint_sha256=source,
        expected_objective_components=("trajectory", "compiler", "preservation"),
        objective_component_hashes=(
            ("trajectory", trajectory),
            ("compiler", compiler),
        ),
        adamw_config_sha256=planned["adamw_config_sha256"],
        fresh_optimizer_identity_sha256=planned["fresh_optimizer_identity_sha256"],
        evaluation_rps=(1.0, 1.10),
        output_root=planned["output_root"],
    )


def _success_receipt(spec: CellSpec) -> CellReceipt:
    checkpoint = _digest("private-checkpoint")
    return CellReceipt(
        cell_key=spec.cell_key,
        shared_evidence=spec.shared_evidence,
        objective_components=spec.expected_objective_components,
        objective_component_hashes=spec.objective_component_hashes,
        adamw_config_sha256=spec.adamw_config_sha256,
        fresh_optimizer_identity_sha256=spec.fresh_optimizer_identity_sha256,
        transaction_id=_digest("transaction")[:32],
        before_transaction_digest=_digest("source-state"),
        after_transaction_digest=_digest("source-state"),
        status="succeeded",
        audits=tuple(
            AuditRef(
                evaluation_rp=rp,
                evaluated_checkpoint_sha256=checkpoint,
                output_path=f"audit/rp-{rp}.jsonl",
                output_sha256=_digest(f"audit:{rp}"),
                row_count=13,
                image_ids=CANONICAL_IMAGE_IDS,
                generation_policy_receipt_sha256=_digest(f"audit-policy:{rp}"),
            )
            for rp in (1.0, 1.10)
        ),
        adamw_proposal_sha256=_digest("proposal"),
        proposal_delta_sha256=_digest("proposal-delta"),
        projection_receipt_sha256=_digest("projection"),
        apply_receipt_sha256=_digest("apply"),
    )


def test_factory_import_and_plan_contract_are_runtime_free(tmp_path) -> None:
    program = """
import json
import sys
from scripts.research import human13_rp_crossover_production as production

print(json.dumps({
    "heavy_imports": [name for name in ("torch", "vllm", "transformers") if name in sys.modules],
    "contract": production.create_node_runtime.plan_contract(),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={"PYTHONPATH": str(REPO_ROOT)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {
        "heavy_imports": [],
        "contract": {
            "schema_version": "human13_rp_crossover_node_runtime_factory.v1",
            "arms": ["A", "B", "C"],
            "evaluation_rps": [1.0, 1.1],
            "max_updates": 1,
            "retry_policy": "none",
            "world_size": 1,
            "requires_user_model_gpu_authority": True,
            "actions": {
                "model_loads": 0,
                "gpu_allocations": 0,
                "subprocess_launches": 0,
                "output_roots_created": 0,
            },
        },
    }
    assert list(tmp_path.iterdir()) == []
    assert (
        runner.inspect_runtime_factory(
            "scripts.research.human13_rp_crossover_production:create_node_runtime"
        )["status"]
        == "execution_ready"
    )


def test_qualification_closes_acquisition_before_real_node_requests_cell_service(
    tmp_path, monkeypatch
) -> None:
    node = _qualification_node(tmp_path)
    spec = _qualification_spec(node)
    events: list[str] = []

    class Composition:
        def acquire_qualification(
            self,
            selected_node: Mapping[str, Any],
            frozen: production.FrozenProductionInputs,
        ) -> production.QualificationAcquisition:
            events.append("acquire")
            assert selected_node["node_id"] == "rp100:qualification"
            assert frozen.manifest_sha256 == production.MANIFEST_SHA256
            component_hashes = dict(spec.objective_component_hashes)
            return production.QualificationAcquisition(
                cell_specs=(spec,),
                source_baselines=tuple(
                    SourceBaselineRef(
                        evaluation_rp=rp,
                        output_a_sha256=_digest(f"source-baseline:{rp}"),
                        output_b_sha256=_digest(f"source-baseline:{rp}"),
                        checkpoint_sha256=(production.SOURCE_CHECKPOINT_PAYLOAD_SHA256),
                        image_ids=CANONICAL_IMAGE_IDS,
                    )
                    for rp in (1.0, 1.10)
                ),
                training_rp=1.0,
                seeds=tuple(range(30001, 30017)),
                image_ids=CANONICAL_IMAGE_IDS,
                native_request_count=208,
                native_batch_count=52,
                manifest_sha256=production.MANIFEST_SHA256,
                tokenizer_sha256=production.TOKENIZER_SHA256,
                prompt_policy_fingerprint=production.PROMPT_POLICY_FINGERPRINT,
                alias_bank_sha256=production.ALIAS_BANK_SHA256,
                nested_objective_hashes={
                    "A": (("trajectory", component_hashes["trajectory"]),),
                    "B": (
                        ("trajectory", component_hashes["trajectory"]),
                        ("compiler", component_hashes["compiler"]),
                    ),
                    "C": spec.objective_component_hashes,
                },
                streaming_mode="per_image_or_pack",
            )

        def close_acquisition(self) -> production.AcquisitionReleaseReceipt:
            events.append("close")
            return production.AcquisitionReleaseReceipt(
                engine_closed=True,
                model_released=True,
                panel_wide_logits_retained=False,
            )

        def services_for_cell(self, selected_spec: CellSpec) -> CellRuntimeServices:
            events.append("services")
            assert selected_spec == spec
            return cast(CellRuntimeServices, object())

    def fake_run_cell(selected_spec, *, services, receipt_writer):
        assert services is not None
        receipt = _success_receipt(selected_spec)
        receipt_writer(receipt)
        events.append("cell")
        return receipt

    monkeypatch.setattr(runner, "run_cell", fake_run_cell)
    factory = runner.declare_node_runtime_factory(
        lambda selected: production.create_node_runtime(
            selected, _composition=Composition()
        )
    )

    terminal = runner._execute_node(
        node,
        receipt_path=node["receipt_path"],
        runtime_factory=factory,
    )

    events.append("terminal")
    assert events == ["acquire", "close", "services", "cell", "terminal"]
    assert terminal["status"] == "succeeded"
    assert [item["cell_key"]["arm_id"] for item in terminal["cell_specs"]] == ["C"]
    assert Path(node["receipt_path"]).is_file()


def test_missing_live_pack_owner_fails_closed_with_a_durable_node_terminal(
    tmp_path,
) -> None:
    node = _qualification_node(tmp_path, rp=1.10)

    with pytest.raises(
        production.ProductionCompositionUnavailable,
        match="sampled-trajectory packed-logit materializer",
    ):
        runner._execute_node(
            node,
            receipt_path=node["receipt_path"],
            runtime_factory=production.create_node_runtime,
        )

    payload = json.loads(Path(node["receipt_path"]).read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["cell_specs"] == []
    assert "ProductionCompositionUnavailable" in payload["failure_reason"]
    assert not Path(node["cells"][0]["output_root"]).exists()


def test_factory_rejects_forged_qualification_seeds_before_acquisition(
    tmp_path,
) -> None:
    node = _qualification_node(tmp_path)
    node["seeds"][0] = 30000

    with pytest.raises(ValueError, match="exact qualification node"):
        production.create_node_runtime(node).acquire_cell_specs(node)

    assert not Path(node["cells"][0]["output_root"]).exists()
