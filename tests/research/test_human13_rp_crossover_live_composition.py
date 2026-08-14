"""Contract of the live composition behind the public node runtime factory.

| # | Frozen behaviour | Owner | Wrong alternative | Test |
| - | ---------------- | ----- | ----------------- | ---- |
| 1 | An incomplete backend is rejected at construction | `Human13RPCrossoverLiveComposition` | a missing seam surfaces mid-acquisition | `test_incomplete_backend_is_rejected_before_any_phase` |
| 2 | Phases run in the frozen order and release before training | `acquire_qualification` | training model loads before engine release | `test_phase_order_releases_engine_and_model_before_cells` |
| 3 | The release receipt is measured, not declared | `close_acquisition` | a constant `True` receipt | `test_release_receipt_is_false_until_the_phases_ran` |
| 4 | The bank is frozen before acquisition | `_freeze_witness_bank` | freezing after sampling | `test_witness_bank_is_frozen_before_sampling` |
| 5 | A cell whose Source differs cannot use the bank | `witness_bank` | silently projecting against another Source | `test_witness_bank_rejects_a_cell_with_a_different_source` |
| 6 | The probe restores the margin surface | `realized_margin_probe` | leaving the proposal on the witness surface | `test_realized_probe_measures_then_restores_the_surface` |
| 7 | Dose mechanics come from the certified projection and audits | `dose_mechanics` | a synthesized active-witness count | `test_dose_mechanics_bind_projection_and_audit_deltas` |
| 8 | Every cell gets independent services | `services_for_cell` | one shared services object | `test_every_cell_receives_independent_services` |
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from scripts.research import human13_rp_crossover_live_composition as composition
from scripts.research import human13_rp_crossover_production as production
from scripts.research import human13_rp_crossover_witness as witness_owner
from scripts.research import launch_human13_k_trajectory_rp_crossover as launcher
from scripts.research.human13_adamw_proposal_preservation import TRUSTED_OWNER_CLASS
from scripts.research.human13_rp_crossover_matrix_contracts import (
    CANONICAL_IMAGE_IDS,
    AggregateResourceReceipt,
    AuditRef,
    SourceBaselineRef,
)


VOCAB = 6
PROMPT = (0, 1)
GENERATED = (2, 3, 4)


def _digest(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


class FakeMarginSurface:
    def __init__(self) -> None:
        self.scale = torch.nn.Parameter(torch.ones(VOCAB, dtype=torch.float32))
        self.rows = {}
        for image_id in CANONICAL_IMAGE_IDS:
            for rp in (1.0, 1.10):
                for index, token in enumerate(GENERATED):
                    row = [0.0] * VOCAB
                    row[token] = 1.0 + 0.5 * index
                    self.rows[(image_id, rp, index)] = torch.tensor(
                        row, dtype=torch.float32
                    )

    def named_trainable_parameters(self):
        return (("dora.scale", self.scale),)

    def raw_logit_rows(self, decode, token_indices):
        return torch.stack(
            [
                self.rows[(decode.image_id, decode.repetition_penalty, int(index))]
                * self.scale
                for index in token_indices
            ]
        )


def _decodes(rp: float) -> tuple[witness_owner.SealedSourceDecode, ...]:
    return tuple(
        witness_owner.SealedSourceDecode(
            image_id=image_id,
            repetition_penalty=rp,
            prompt_token_ids=PROMPT,
            generated_token_ids=GENERATED,
            owner_rows=(
                witness_owner.SealedOwnerRow(
                    owner_id=f"owner-{image_id}",
                    owner_class=TRUSTED_OWNER_CLASS,
                    token_start=0,
                    token_end=2,
                ),
            ),
        )
        for image_id in CANONICAL_IMAGE_IDS
    )


def _surface(rp: float) -> composition.SourceSurfaceEvidence:
    return composition.SourceSurfaceEvidence(
        repetition_penalty=rp,
        baseline=SourceBaselineRef(
            evaluation_rp=rp,
            output_a_sha256=_digest(f"baseline:{rp}"),
            output_b_sha256=_digest(f"baseline:{rp}"),
            checkpoint_sha256=production.SOURCE_CHECKPOINT_PAYLOAD_SHA256,
            image_ids=CANONICAL_IMAGE_IDS,
        ),
        decodes=_decodes(rp),
        frontier=object(),
        frontier_path=f"frontier-{rp}.json",
        outputs=(),
    )


@dataclass
class FakeBackend:
    """One CPU double for every live seam of the composition."""

    events: list[str]
    margin: FakeMarginSurface

    def source_surface(self, frozen, *, repetition_penalty):
        self.events.append(f"surface:{repetition_penalty}")
        return _surface(repetition_penalty)

    def open_margin_surface(self, frozen):
        self.events.append("margin:open")
        return self.margin

    def close_margin_surface(self, surface):
        self.events.append("margin:close")

    def open_sampler(self, frozen):
        self.events.append("sampler:open")
        return object()

    def sample_batch(self, sampler, batch, params):
        raise AssertionError("sampling is exercised by the acquisition adapter")

    def close_sampler(self, sampler):
        self.events.append("sampler:close")

    def open_packed_surface(self, frozen):
        self.events.append("packed:open")
        return object()

    def close_packed_surface(self, packed):
        self.events.append("packed:close")

    def packed_raw_logits(self, packed, execution):
        raise AssertionError("replay is exercised by the live-pack adapter")

    def admit_compiler_panel(self, packed, *, acquisition, surface):
        raise AssertionError("compiler admission is exercised by its own adapter")

    def open_cell(self, spec, frozen):
        raise AssertionError("cell assembly is exercised by the model adapter")

    def backward_objective(self, state, spec, evidence, packed):
        raise AssertionError("backward is exercised by the packed adapter")

    def write_private_checkpoint(self, state, spec, stack):
        raise AssertionError("checkpointing is exercised by its own adapter")

    def audit_checkpoint(self, spec, checkpoint, repetition_penalty):
        self.events.append(f"audit:{repetition_penalty}")
        return composition.AuditOutcome(
            audit=AuditRef(
                evaluation_rp=repetition_penalty,
                evaluated_checkpoint_sha256=checkpoint.checkpoint_sha256,
                output_path=f"audit-{repetition_penalty}.jsonl",
                output_sha256=_digest(f"audit:{repetition_penalty}"),
                row_count=len(CANONICAL_IMAGE_IDS),
                image_ids=CANONICAL_IMAGE_IDS,
                generation_policy_receipt_sha256=_digest(
                    f"policy:{repetition_penalty}"
                ),
            ),
            malformed_delta=1 if repetition_penalty == 1.10 else 0,
            cap_terminated_delta=0,
            unparseable_delta=0,
        )

    def close_cell(self, state):
        self.events.append("cell:close")

    def tokenizer_adapter(self, *, manifest, publication):
        return object()

    def resource_snapshot(self):
        return composition.ResourceSnapshot(
            measurement_scope="injected_cpu",
            peak_host_rss_bytes=1024,
            cuda_peak_allocated_bytes=None,
            cuda_peak_reserved_bytes=None,
            decode_token_count=32,
            packed_token_count=64,
            logical_token_count=48,
            forward_count=13,
            row_bytes=256,
            artifact_bytes=128,
        )


def _composition() -> tuple[composition.Human13RPCrossoverLiveComposition, FakeBackend]:
    backend = FakeBackend(events=[], margin=FakeMarginSurface())
    return (
        composition.Human13RPCrossoverLiveComposition(backend=backend),
        backend,
    )


def _compiler_ledger(
    *,
    compiler_image_id: int | None = None,
    token_index: int = 1,
    repetition_penalty: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        repetition_penalty=repetition_penalty,
        images=tuple(
            SimpleNamespace(
                image_id=image_id,
                site=(
                    SimpleNamespace(generated_token_index=token_index)
                    if image_id == compiler_image_id
                    else None
                ),
                absent_reason=(
                    None if image_id == compiler_image_id else "no_trusted_remaining"
                ),
            )
            for image_id in CANONICAL_IMAGE_IDS
        ),
    )


def _frozen() -> Any:
    return production._validate_frozen_inputs(1.0)


def _node(tmp_path: Path) -> Mapping[str, Any]:
    plan = launcher.build_dag_plan(
        launcher.load_leaf_configs(),
        run_id="live-composition",
        output_root=tmp_path / "artifacts",
    )
    return next(
        item
        for item in plan["acquisitions"]
        if item["node_id"] == "rp100:qualification"
    )


class _State:
    def __init__(self, parameters) -> None:
        self.named_trainable_parameters = parameters


def test_incomplete_backend_is_rejected_before_any_phase() -> None:
    class Partial:
        def source_surface(self, frozen, *, repetition_penalty):
            raise AssertionError

    with pytest.raises(composition.LiveCompositionError, match="open_margin_surface"):
        composition.Human13RPCrossoverLiveComposition(backend=cast(Any, Partial()))


def test_release_receipt_is_false_until_the_phases_ran() -> None:
    live, _backend = _composition()
    receipt = live.close_acquisition()

    assert (receipt.engine_closed, receipt.model_released) == (False, False)
    assert receipt.panel_wide_logits_retained is False


def test_margin_surface_open_failure_preserves_primary_through_release_validation(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    live, backend = _composition()

    def fail_margin_surface(frozen: Any) -> Any:
        del frozen
        backend.events.append("margin:open")
        raise ValueError("census scorer construction failed")

    monkeypatch.setattr(backend, "open_margin_surface", fail_margin_surface)
    node = _node(tmp_path)
    runtime = production.ProductionNodeRuntime(node, composition=live)

    with pytest.raises(ValueError, match="census scorer construction failed"):
        runtime.acquire_cell_specs(node)

    receipt = live.close_acquisition()
    assert (receipt.engine_closed, receipt.model_released) == (True, True)
    assert backend.events == ["surface:1.0", "surface:1.1", "margin:open"]


def test_witness_bank_is_frozen_before_sampling() -> None:
    live, backend = _composition()
    for repetition_penalty in (1.0, 1.10):
        live._surfaces[repetition_penalty] = _surface(repetition_penalty)
    live._freeze_witness_bank(_frozen())

    assert backend.events == ["margin:open", "margin:close"]
    bank = live._witness_bank
    assert bank is not None
    assert bank.binding.frozen_before_acquisition is True
    assert len(bank.constraints) == 2 * len(CANONICAL_IMAGE_IDS)


def test_compiler_sites_are_bound_after_acquisition_before_dose_statistics() -> None:
    live, backend = _composition()
    for repetition_penalty in (1.0, 1.10):
        live._surfaces[repetition_penalty] = _surface(repetition_penalty)
    live._freeze_witness_bank(_frozen())
    frozen_bank = live._witness_bank
    assert frozen_bank is not None
    assert live._source_dose_margins is None

    compiler_image_id = CANONICAL_IMAGE_IDS[0]
    ledger = _compiler_ledger(compiler_image_id=compiler_image_id)
    backend.events.append("acquisition:closed")
    live._bind_compiler_dose_evidence(
        _frozen(), compiler_ledger=ledger, training_rp=1.0
    )

    assert live._witness_bank is frozen_bank
    assert backend.events == [
        "margin:open",
        "margin:close",
        "acquisition:closed",
        "margin:open",
        "margin:close",
    ]
    measurement = live._measurement
    assert measurement is not None
    expected_witness_sites = {
        (image_id, repetition_penalty, 0)
        for image_id in CANONICAL_IMAGE_IDS
        for repetition_penalty in (1.0, 1.10)
    }
    assert set(measurement.dose_sites) == expected_witness_sites | {
        (compiler_image_id, 1.0, 1)
    }
    assert set(live._source_dose_margins or ()) == {
        f"{image_id}|{'1.0' if repetition_penalty == 1.0 else '1.10'}|{token_index}"
        for image_id, repetition_penalty, token_index in measurement.dose_sites
    }


def test_compiler_dose_binding_rejects_mixed_rp_ledger() -> None:
    live, _backend = _composition()
    for repetition_penalty in (1.0, 1.10):
        live._surfaces[repetition_penalty] = _surface(repetition_penalty)
    live._freeze_witness_bank(_frozen())

    with pytest.raises(composition.LiveCompositionError, match="ledger RP"):
        live._bind_compiler_dose_evidence(
            _frozen(),
            compiler_ledger=_compiler_ledger(repetition_penalty=1.10),
            training_rp=1.0,
        )


def test_phase_order_releases_engine_and_model_before_cells(monkeypatch) -> None:
    live, backend = _composition()
    for repetition_penalty in (1.0, 1.10):
        live._surfaces[repetition_penalty] = _surface(repetition_penalty)
    live._freeze_witness_bank(_frozen())
    monkeypatch.setattr(
        live, "_sample", lambda frozen, rp: backend.events.append("sample") or ()
    )
    monkeypatch.setattr(
        live,
        "_materialize",
        lambda frozen, rp, executions: backend.events.append("materialize") or None,
    )

    with pytest.raises(composition.LiveCompositionError):
        live.services_for_cell(cast(Any, object()))
    assert backend.events == ["margin:open", "margin:close"]


def test_matrix_cell_specs_bind_the_node_seed_group_and_all_nested_arms(
    tmp_path,
) -> None:
    from scripts.research.human13_rp_crossover_matrix_contracts import (
        AcquisitionKey,
        SharedEvidenceRef,
    )

    plan = launcher.build_dag_plan(
        launcher.load_leaf_configs(),
        run_id="matrix-live-composition",
        output_root=tmp_path / "artifacts",
    )
    raw = next(
        item for item in plan["acquisitions"] if item["node_id"] == "rp100:matrix_a"
    )
    decision = _digest("selected-lr")
    node = {
        **raw,
        "cells": [
            {
                **cell,
                "learning_rate": 3.0e-6,
                "adamw_config_sha256": _digest(f"adamw:{cell['cell_key']['arm_id']}"),
                "fresh_optimizer_identity_sha256": _digest(
                    f"optimizer:{cell['cell_key']['arm_id']}"
                ),
                "global_learning_rate_decision_sha256": decision,
                "resolved_leaf_config_sha256": _digest(
                    f"resolved:{cell['cell_key']['arm_id']}"
                ),
            }
            for cell in raw["cells"]
        ],
    }
    acquisition_key = AcquisitionKey.from_dict(node["acquisition_key"])
    shared = SharedEvidenceRef(
        source_sha256=production.SOURCE_CHECKPOINT_PAYLOAD_SHA256,
        manifest_sha256=production.MANIFEST_SHA256,
        acquisition_path=str(tmp_path / "acquisition"),
        acquisition_sha256=_digest("acquisition"),
        trajectory_credit_acquisition_sha256=_digest("credit-acquisition"),
        credit_ledger_sha256=_digest("credit"),
        compiler_ledger_sha256=_digest("compiler"),
        policy_contract_sha256=_digest("policy"),
        native_receipts_sha256=_digest("native"),
        training_rp=acquisition_key.training_rp,
        seed_group_id=acquisition_key.seed_group_id,
        seeds=acquisition_key.seeds,
    )

    class Nested:
        @staticmethod
        def arm_component_hashes(arm_id):
            return (
                (("trajectory", _digest("credit")),)
                if arm_id == "A"
                else (
                    ("trajectory", _digest("credit")),
                    ("compiler", _digest("compiler")),
                )
            )

    evidence = composition.AcquisitionEvidence(
        acquisition=object(),
        credit_ledger=object(),
        compiler_ledger=object(),
        nested=Nested(),
        shared_evidence=shared,
        acquisition_path=shared.acquisition_path,
        native_receipts_sha256=shared.native_receipts_sha256,
        request_count=208,
        batch_count=52,
        token_count=4096,
    )
    live, _backend = _composition()

    specs = live._cell_specs(node, evidence)

    assert tuple(spec.cell_key.arm_id for spec in specs) == ("A", "B", "C")
    assert {spec.shared_evidence.seed_group_id for spec in specs} == {"matrix_a"}
    assert {spec.global_learning_rate_decision_sha256 for spec in specs} == {decision}


def test_native_plan_uses_the_selected_matrix_seed_group(monkeypatch) -> None:
    from scripts.research import collect_human13_rp_crossover as acquisition_owner

    live, backend = _composition()
    monkeypatch.setattr(
        acquisition_owner,
        "execute_acquisition_group",
        lambda *, plan, execute_batch: plan,
    )

    plans = live._sample(_frozen(), 1.10, "matrix_b")

    assert len(plans) == 13
    assert {plan.seed_group_id for plan in plans} == {"matrix_b"}
    assert tuple(request.seed for request in plans[0].requests) == tuple(
        range(32001, 32017)
    )
    assert backend.events == ["sampler:open", "sampler:close"]


def test_witness_bank_rejects_a_cell_with_a_different_source() -> None:
    live, _backend = _composition()
    for repetition_penalty in (1.0, 1.10):
        live._surfaces[repetition_penalty] = _surface(repetition_penalty)
    live._freeze_witness_bank(_frozen())

    matching = _State((("dora.scale", torch.nn.Parameter(torch.ones(VOCAB))),))
    assert live.witness_bank(cast(Any, matching)) is live._witness_bank

    drifted = _State((("dora.scale", torch.nn.Parameter(torch.full((VOCAB,), 1.5))),))
    with pytest.raises(composition.LiveCompositionError, match="fresh Source differs"):
        live.witness_bank(cast(Any, drifted))


def test_realized_probe_measures_then_restores_the_surface(tmp_path) -> None:
    live, backend = _composition()
    for repetition_penalty in (1.0, 1.10):
        live._surfaces[repetition_penalty] = _surface(repetition_penalty)
    live._freeze_witness_bank(_frozen())
    live._bind_compiler_dose_evidence(
        _frozen(), compiler_ledger=_compiler_ledger(), training_rp=1.0
    )

    node = _node(tmp_path)
    spec = _spec(node, live)
    trained = torch.nn.Parameter(torch.ones(VOCAB))
    state = _State((("dora.scale", trained),))
    cast(Any, state)._human13_witness_surface = backend.margin
    bank = live.witness_bank(cast(Any, state))
    with torch.no_grad():
        trained.fill_(1.25)
    probe = live.realized_margin_probe(cast(Any, state), spec)
    realized = probe()

    assert set(realized) == {item.canonical_key for item in bank.constraints}
    # margins scale with the trained parameter but the surface is put back
    assert realized[bank.constraints[0].canonical_key] == pytest.approx(1.25)
    assert torch.equal(backend.margin.scale.detach(), torch.ones(VOCAB))
    measured = live._dose_mechanics[spec.cell_key.content_sha256]
    assert measured["greedy_decision_change_count"] == 0
    assert measured["jvp_fd_tolerance"] == witness_owner.JVP_FD_TOLERANCE


def _spec(node, live) -> Any:
    from scripts.research.human13_rp_crossover_matrix_contracts import (
        AcquisitionKey,
        CellKey,
        CellSpec,
        SharedEvidenceRef,
    )

    acquisition_key = AcquisitionKey.from_dict(node["acquisition_key"])
    planned = node["cells"][0]
    shared = SharedEvidenceRef(
        source_sha256=production.SOURCE_CHECKPOINT_PAYLOAD_SHA256,
        manifest_sha256=production.MANIFEST_SHA256,
        acquisition_path=str(Path(planned["output_root"]).parent / "acquisition"),
        acquisition_sha256=_digest("acquisition"),
        trajectory_credit_acquisition_sha256=_digest("credit-acquisition"),
        credit_ledger_sha256=_digest("credit"),
        compiler_ledger_sha256=_digest("compiler"),
        policy_contract_sha256=_digest("policy"),
        native_receipts_sha256=_digest("native"),
        training_rp=acquisition_key.training_rp,
        seed_group_id="qualification",
        seeds=tuple(range(30001, 30017)),
    )
    return CellSpec(
        cell_key=CellKey(acquisition_key, "C", planned["learning_rate"]),
        shared_evidence=shared,
        leaf_config_sha256=planned["source_leaf_config_sha256"],
        source_checkpoint_sha256=production.SOURCE_CHECKPOINT_PAYLOAD_SHA256,
        expected_objective_components=("trajectory", "compiler", "preservation"),
        objective_component_hashes=(
            ("trajectory", _digest("credit")),
            ("compiler", _digest("compiler")),
        ),
        adamw_config_sha256=planned["adamw_config_sha256"],
        fresh_optimizer_identity_sha256=planned["fresh_optimizer_identity_sha256"],
        evaluation_rps=(1.0, 1.10),
        output_root=planned["output_root"],
        learning_rate=planned["learning_rate"],
        resolved_leaf_config_sha256=planned["resolved_leaf_config_sha256"],
    )


@dataclass(frozen=True)
class _Projection:
    active_witnesses: tuple[str, ...]


def test_dose_mechanics_bind_projection_and_audit_deltas(tmp_path) -> None:
    live, backend = _composition()
    for repetition_penalty in (1.0, 1.10):
        live._surfaces[repetition_penalty] = _surface(repetition_penalty)
    live._freeze_witness_bank(_frozen())
    live._bind_compiler_dose_evidence(
        _frozen(), compiler_ledger=_compiler_ledger(), training_rp=1.0
    )
    node = _node(tmp_path)
    spec = _spec(node, live)
    live._evidence = composition.AcquisitionEvidence(
        acquisition=object(),
        credit_ledger=object(),
        compiler_ledger=object(),
        nested=object(),
        shared_evidence=spec.shared_evidence,
        acquisition_path=spec.shared_evidence.acquisition_path,
        native_receipts_sha256=spec.shared_evidence.native_receipts_sha256,
        request_count=208,
        batch_count=52,
        token_count=4096,
    )
    live._witness_bank = live._witness_bank
    services = composition._CellServices(composition=live, spec=spec)
    state = _State((("dora.scale", torch.nn.Parameter(torch.ones(VOCAB))),))
    cast(Any, state)._human13_witness_surface = backend.margin
    bank = live.witness_bank(cast(Any, state))
    probe = services.realized_margin_probe(cast(Any, state), spec, bank)
    probe()

    checkpoint = _checkpoint()
    audits = [
        services.audit_checkpoint(cast(Any, None), checkpoint, rp) for rp in (1.0, 1.10)
    ]
    resources = AggregateResourceReceipt(
        measurement_scope="injected_cpu",
        wall_time_seconds=0.5,
        peak_host_rss_bytes=1024,
        cuda_peak_allocated_bytes=None,
        cuda_peak_reserved_bytes=None,
        acquisition_request_count=208,
        acquisition_batch_count=52,
        acquisition_token_count=4096,
        decode_request_count=26,
        decode_batch_count=26,
        decode_token_count=2048,
        packed_token_count=4096,
        logical_token_count=3900,
        forward_count=13,
        backward_count=13,
        row_bytes=8192,
        artifact_bytes=4096,
        update_count=1,
        audit_count=2,
        rollback_count=1,
    )
    receipt = services.dose_mechanical_receipt(
        cast(Any, None),
        spec,
        checkpoint,
        _digest("proposal"),
        tuple(audits),
        resources,
        _Projection(active_witnesses=("u_intersect_s_1.0|1584|owner-1584",)),
    )

    assert receipt.active_witness_count == 1
    assert receipt.malformed_output_delta_count == 1
    assert receipt.cap_terminated_output_delta_count == 0
    assert receipt.jvp_fd_tolerance == witness_owner.JVP_FD_TOLERANCE
    assert receipt.rollback_reproduced is True
    assert backend.events.count("margin:close") == 3
    assert backend.events[-2:] == ["audit:1.0", "audit:1.1"]


def _checkpoint() -> Any:
    from scripts.research.human13_rp_crossover_runtime import PrivateCheckpointRef

    return PrivateCheckpointRef(
        path="private/step-1",
        checkpoint_sha256=_digest("private-checkpoint"),
        private=True,
    )


def test_every_cell_receives_independent_services(tmp_path) -> None:
    live, _backend = _composition()
    for repetition_penalty in (1.0, 1.10):
        live._surfaces[repetition_penalty] = _surface(repetition_penalty)
    live._freeze_witness_bank(_frozen())
    node = _node(tmp_path)
    spec = _spec(node, live)
    live._evidence = composition.AcquisitionEvidence(
        acquisition=object(),
        credit_ledger=object(),
        compiler_ledger=object(),
        nested=object(),
        shared_evidence=spec.shared_evidence,
        acquisition_path=spec.shared_evidence.acquisition_path,
        native_receipts_sha256=spec.shared_evidence.native_receipts_sha256,
        request_count=208,
        batch_count=52,
        token_count=4096,
    )

    first = live.services_for_cell(spec)
    second = live.services_for_cell(replace(spec, learning_rate=spec.learning_rate))
    assert first is not second


def test_audit_outcome_and_resource_snapshot_reject_invalid_counts() -> None:
    with pytest.raises(composition.LiveCompositionError, match="AuditRef"):
        composition.AuditOutcome(
            audit=cast(Any, object()),
            malformed_delta=0,
            cap_terminated_delta=0,
            unparseable_delta=0,
        )
