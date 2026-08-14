#!/usr/bin/env python3
"""The live composition behind the public Human-13 RP-crossover node factory.

This module owns only order, admission, and receipts.  Every scientific step is
delegated to the owner that already implements it: native batch-four sampling
and replay parity (:mod:`collect_human13_rp_crossover`), packed materialization
(:mod:`human13_rp_crossover_live_packs`), row credit
(:mod:`human13_trajectory_credit`), the sparse compiler
(:mod:`human13_greedy_compiler`), the exact AdamW proposal and its projection
(:mod:`human13_adamw_proposal_preservation`), the frozen witness and dose
mechanics (:mod:`human13_rp_crossover_witness`), model assembly
(:mod:`human13_live_model`), and clean-greedy audits
(:mod:`human13_live_eval`).

The frozen phase order is:

1. two deterministic Source clean-greedy surfaces (RP 1.0 and RP 1.10);
2. the owner-wise witness bank, frozen before any acquisition;
3. one native batch-four K16 acquisition, then engine release;
4. packed replay, parity publication, credit/compiler ledgers, then model
   release -- both before any training model load;
5. independent per-dose cell runtime services.

Import is standard-library plus Torch only.  Every live seam is reached through
:class:`LiveCompositionBackend`, so the composition contract is CPU-testable
while the default backend performs the real work.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Protocol, cast

import torch

from scripts.research.human13_adamw_proposal_preservation import (
    LEGACY_M_OWNER_CLASS,
    TRUSTED_OWNER_CLASS,
    FrozenWitnessBank,
    ParameterLayout,
    WitnessBinding,
)
from scripts.research.human13_rp_crossover_matrix_contracts import (
    CANONICAL_IMAGE_IDS,
    EVALUATION_RPS,
    PROPOSAL_COMPONENTS_BY_ARM,
    AcquisitionKey,
    AggregateResourceReceipt,
    AuditRef,
    CellKey,
    CellSpec,
    DoseMechanicalReceipt,
    SharedEvidenceRef,
    SourceBaselineRef,
)
from scripts.research.human13_rp_crossover_runtime import (
    CellExecutionState,
    ObjectiveBackwardReceipt,
    PrivateCheckpointRef,
)
from scripts.research import human13_rp_crossover_witness as witness_owner


SCHEMA_VERSION = "human13_rp_crossover_live_composition.v1"
UNIT_ID = "2026-08-14-human13-k-trajectory-rp-crossover-screen"
QUALIFICATION_SEED_GROUP = "qualification"
NATIVE_REQUEST_COUNT = 208
NATIVE_BATCH_COUNT = 52
STREAMING_MODE = "per_image_or_pack"


class LiveCompositionError(RuntimeError):
    """Raised when the live composition cannot be admitted fail-closed."""


@dataclass(frozen=True)
class SourceSurfaceEvidence:
    """One RP-specific sealed Source clean-greedy surface and its baseline."""

    repetition_penalty: float
    baseline: SourceBaselineRef
    decodes: tuple[witness_owner.SealedSourceDecode, ...]
    frontier: Any
    frontier_path: str
    outputs: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if self.repetition_penalty not in EVALUATION_RPS:
            raise LiveCompositionError("Source surface RP is outside the sealed pair")
        if not isinstance(self.baseline, SourceBaselineRef) or (
            self.baseline.evaluation_rp != self.repetition_penalty
        ):
            raise LiveCompositionError("Source baseline RP differs from its surface")
        image_ids = tuple(item.image_id for item in self.decodes)
        if image_ids != CANONICAL_IMAGE_IDS:
            raise LiveCompositionError(
                "sealed Source surface must cover the exact canonical panel order"
            )
        if any(
            item.repetition_penalty != self.repetition_penalty for item in self.decodes
        ):
            raise LiveCompositionError("sealed Source decode RP differs from surface")


@dataclass(frozen=True)
class AcquisitionEvidence:
    """The published acquisition and its byte-identical nested objectives."""

    acquisition: Any
    credit_ledger: Any
    compiler_ledger: Any
    nested: Any
    shared_evidence: SharedEvidenceRef
    acquisition_path: str
    native_receipts_sha256: str
    request_count: int
    batch_count: int
    token_count: int


@dataclass(frozen=True)
class AuditOutcome:
    """One free-running clean-greedy audit and its nonnegative output deltas."""

    audit: AuditRef
    malformed_delta: int
    cap_terminated_delta: int
    unparseable_delta: int

    def __post_init__(self) -> None:
        if not isinstance(self.audit, AuditRef):
            raise LiveCompositionError("an audit outcome requires a typed AuditRef")
        for field in (
            "malformed_delta",
            "cap_terminated_delta",
            "unparseable_delta",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LiveCompositionError(
                    f"audit {field} must be a nonnegative integer"
                )


@dataclass(frozen=True)
class ResourceSnapshot:
    """Measured counters the runtime cannot derive from its typed receipts."""

    measurement_scope: str
    peak_host_rss_bytes: int
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None
    decode_token_count: int
    packed_token_count: int
    logical_token_count: int
    forward_count: int
    row_bytes: int
    artifact_bytes: int


class LiveCompositionBackend(Protocol):
    """Every seam that needs a model, an engine, or the filesystem."""

    def source_surface(
        self, frozen: Any, *, repetition_penalty: float
    ) -> SourceSurfaceEvidence: ...

    def open_margin_surface(self, frozen: Any) -> Any: ...

    def close_margin_surface(self, surface: Any) -> None: ...

    def open_sampler(self, frozen: Any) -> Any: ...

    def sample_batch(
        self, sampler: Any, batch: Any, params: tuple[Any, ...]
    ) -> Any: ...

    def close_sampler(self, sampler: Any) -> None: ...

    def open_packed_surface(self, frozen: Any) -> Any: ...

    def close_packed_surface(self, packed: Any) -> None: ...

    def packed_raw_logits(self, packed: Any, execution: Any) -> Any: ...

    def admit_compiler_panel(
        self, packed: Any, *, acquisition: Any, surface: SourceSurfaceEvidence
    ) -> Any: ...

    def open_cell(self, spec: CellSpec, frozen: Any) -> CellExecutionState: ...

    def backward_objective(
        self,
        state: CellExecutionState,
        spec: CellSpec,
        evidence: AcquisitionEvidence,
        packed: Any,
    ) -> ObjectiveBackwardReceipt: ...

    def write_private_checkpoint(
        self, state: CellExecutionState, spec: CellSpec, stack: ExitStack
    ) -> PrivateCheckpointRef: ...

    def audit_checkpoint(
        self,
        spec: CellSpec,
        checkpoint: PrivateCheckpointRef,
        repetition_penalty: float,
    ) -> AuditOutcome: ...

    def close_cell(self, state: CellExecutionState) -> None: ...

    def tokenizer_adapter(self, *, manifest: Any, publication: Any) -> Any: ...

    def resource_snapshot(self) -> ResourceSnapshot: ...


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _flat_float64(
    named_parameters: Sequence[tuple[str, torch.Tensor]], layout: ParameterLayout
) -> torch.Tensor:
    """Flatten one parameter surface in exactly the frozen layout order."""

    values = dict(named_parameters)
    entries = layout.entries
    if set(values) != {entry.name for entry in entries} or len(values) != len(entries):
        raise LiveCompositionError(
            "parameter surface names differ from the frozen layout"
        )
    chunks = []
    for entry in entries:
        tensor = values[entry.name]
        if tuple(tensor.shape) != entry.shape:
            raise LiveCompositionError(
                f"parameter {entry.name} shape differs from the frozen layout"
            )
        chunks.append(tensor.detach().reshape(-1).to(dtype=torch.float64))
    return torch.cat(chunks)


class Human13RPCrossoverLiveComposition:
    """One node's exact acquisition owner and per-cell runtime services owner."""

    def __init__(self, *, backend: LiveCompositionBackend) -> None:
        for name in (
            "source_surface",
            "open_margin_surface",
            "close_margin_surface",
            "open_sampler",
            "sample_batch",
            "close_sampler",
            "open_packed_surface",
            "close_packed_surface",
            "packed_raw_logits",
            "admit_compiler_panel",
            "open_cell",
            "backward_objective",
            "write_private_checkpoint",
            "audit_checkpoint",
            "close_cell",
            "tokenizer_adapter",
            "resource_snapshot",
        ):
            if not callable(getattr(backend, name, None)):
                raise LiveCompositionError(
                    f"live composition backend does not expose {name}()"
                )
        self._backend = backend
        self._frozen: Any | None = None
        self._training_rp: float | None = None
        self._surfaces: dict[float, SourceSurfaceEvidence] = {}
        self._measurement: witness_owner.WitnessMeasurement | None = None
        self._witness_bank: FrozenWitnessBank | None = None
        self._source_flat: torch.Tensor | None = None
        self._source_dose_margins: Mapping[str, float] | None = None
        self._margin_surface: Any | None = None
        self._evidence: AcquisitionEvidence | None = None
        self._acquisition_root_path: Path | None = None
        self._packed: Any | None = None
        self._engine_closed = False
        self._model_released = False
        self._panel_wide_logits_retained = False
        self._dose_mechanics: dict[str, dict[str, Any]] = {}

    # -- acquisition owner ---------------------------------------------------

    def acquire_qualification(self, node: Mapping[str, Any], frozen: Any) -> Any:
        """Run the frozen phase order and publish one typed acquisition."""

        from scripts.research.human13_rp_crossover_production import (
            QualificationAcquisition,
        )

        self._frozen = frozen
        training_rp = float(node["training_rp"])
        self._training_rp = training_rp
        acquisition_key = AcquisitionKey.from_dict(node["acquisition_key"])
        if (
            acquisition_key.training_rp != training_rp
            or tuple(node["seeds"]) != acquisition_key.seeds
        ):
            raise LiveCompositionError(
                "node acquisition policy differs from its sealed acquisition key"
            )
        roots = {Path(cell["output_root"]).parent for cell in node["cells"]}
        if len(roots) != 1:
            raise LiveCompositionError(
                "one qualification node must own exactly one vertical root"
            )
        self._acquisition_root_path = roots.pop() / "acquisition"

        # 1. both sealed Source surfaces: baselines, frontier, and owner rows
        for repetition_penalty in EVALUATION_RPS:
            surface = self._backend.source_surface(
                frozen, repetition_penalty=repetition_penalty
            )
            if not isinstance(surface, SourceSurfaceEvidence):
                raise LiveCompositionError(
                    "Source surface owner must return typed SourceSurfaceEvidence"
                )
            self._surfaces[repetition_penalty] = surface

        # 2. the owner-wise witness bank, frozen before acquisition
        self._freeze_witness_bank(frozen)

        # 3. native batch-four K16 acquisition, then engine release
        executions = self._sample(frozen, training_rp, acquisition_key.seed_group_id)

        # 4. packed replay, publication, ledgers, then model release
        evidence = self._materialize(frozen, training_rp, executions)
        self._evidence = evidence

        specs = self._cell_specs(node, evidence)
        return QualificationAcquisition(
            cell_specs=specs,
            source_baselines=tuple(
                self._surfaces[repetition_penalty].baseline
                for repetition_penalty in EVALUATION_RPS
            ),
            training_rp=training_rp,
            seeds=tuple(node["seeds"]),
            image_ids=CANONICAL_IMAGE_IDS,
            native_request_count=evidence.request_count,
            native_batch_count=evidence.batch_count,
            manifest_sha256=frozen.manifest_sha256,
            tokenizer_sha256=frozen.tokenizer_sha256,
            prompt_policy_fingerprint=frozen.prompt_policy_fingerprint,
            alias_bank_sha256=frozen.alias_bank_sha256,
            nested_objective_hashes={
                "A": tuple(evidence.nested.arm_component_hashes("A")),
                "B": tuple(evidence.nested.arm_component_hashes("B")),
                "C": tuple(evidence.nested.arm_component_hashes("C")),
            },
            streaming_mode=STREAMING_MODE,
        )

    def close_acquisition(self) -> Any:
        """Release the engine and every acquisition-phase model exactly once."""

        from scripts.research.human13_rp_crossover_production import (
            AcquisitionReleaseReceipt,
        )

        if self._margin_surface is not None:
            self._backend.close_margin_surface(self._margin_surface)
            self._margin_surface = None
        return AcquisitionReleaseReceipt(
            engine_closed=self._engine_closed,
            model_released=self._model_released,
            panel_wide_logits_retained=self._panel_wide_logits_retained,
        )

    # -- phases --------------------------------------------------------------

    def _freeze_witness_bank(self, frozen: Any) -> None:
        surface = self._backend.open_margin_surface(frozen)
        self._margin_surface = surface
        try:
            measurement = witness_owner.WitnessMeasurement(
                decodes=self._sealed_decodes(),
                surface=surface,
            )
            binding = WitnessBinding(
                unit_id=UNIT_ID,
                source_checkpoint_sha256=frozen.source_checkpoint_payload_sha256,
                manifest_sha256=frozen.manifest_sha256,
                frozen_before_acquisition=True,
            )
            bank = measurement.freeze_witness_bank(binding=binding)
            if measurement.teacher_forced_greedy_change_count() != 0:
                raise LiveCompositionError(
                    "the sealed Source surface is not its own teacher-forced argmax"
                )
            self._measurement = measurement
            self._witness_bank = bank
            self._source_flat = _flat_float64(
                surface.named_trainable_parameters(), bank.layout
            )
            self._source_dose_margins = dict(measurement.dose_site_margins())
        finally:
            # The pre-acquisition witness model never overlaps the vLLM
            # sampler.  Cells reopen a fresh Source witness surface and must
            # reproduce this bank byte-for-byte before its provider is used.
            self._backend.close_margin_surface(surface)
            self._margin_surface = None

    def _sealed_decodes(self) -> tuple[witness_owner.SealedSourceDecode, ...]:
        return tuple(
            decode
            for repetition_penalty in EVALUATION_RPS
            for decode in self._surfaces[repetition_penalty].decodes
        )

    def _sample(
        self, frozen: Any, training_rp: float, seed_group_id: str
    ) -> tuple[Any, ...]:
        from scripts.research.collect_human13_rp_crossover import (
            execute_acquisition_group,
            plan_panel_acquisition,
        )

        plans = plan_panel_acquisition(
            repetition_penalty=training_rp, seed_group_id=seed_group_id
        )
        sampler = self._backend.open_sampler(frozen)
        executions: list[Any] = []
        try:
            for plan in plans:
                executions.append(
                    execute_acquisition_group(
                        plan=plan,
                        execute_batch=(
                            lambda batch, params: self._backend.sample_batch(
                                sampler, batch, params
                            )
                        ),
                    )
                )
        finally:
            self._backend.close_sampler(sampler)
            self._engine_closed = True
        return tuple(executions)

    def _materialize(
        self, frozen: Any, training_rp: float, executions: Sequence[Any]
    ) -> AcquisitionEvidence:
        from scripts.research.collect_human13_rp_crossover import (
            replay_acquisition_group,
        )
        from scripts.research.human13_greedy_compiler import (
            build_compiler_ledger,
            build_nested_arm_artifacts,
        )
        from scripts.research.human13_trajectory_credit import (
            TrajectoryCreditPanelAcquisition,
        )

        packed = self._backend.open_packed_surface(frozen)
        self._packed = packed
        try:
            publications = []
            for execution in executions:
                raw = self._backend.packed_raw_logits(packed, execution)
                replayed, parity = replay_acquisition_group(
                    sampled=execution.group, packed=raw
                )
                publications.append(self._publish(frozen, execution, replayed, parity))
            acquisition = TrajectoryCreditPanelAcquisition(tuple(publications))
            credit_ledger = self._credit_ledger(frozen, acquisition)
            compiler_panel = self._backend.admit_compiler_panel(
                packed,
                acquisition=acquisition,
                surface=self._surfaces[training_rp],
            )
            compiler_ledger = build_compiler_ledger(
                self._manifest(frozen), compiler_panel, credit_ledger
            )
            nested = build_nested_arm_artifacts(
                acquisition=acquisition,
                trajectory_credit_ledger=credit_ledger,
                compiler_ledger=compiler_ledger,
            )
        finally:
            self._backend.close_packed_surface(packed)
            self._model_released = True
        return self._acquisition_evidence(
            frozen=frozen,
            training_rp=training_rp,
            acquisition=acquisition,
            credit_ledger=credit_ledger,
            compiler_ledger=compiler_ledger,
            nested=nested,
            executions=executions,
        )

    def _publish(self, frozen: Any, execution: Any, replayed: Any, parity: Any) -> Any:
        from scripts.research.collect_human13_rp_crossover import (
            load_published_acquisition,
            publish_acquisition_group,
        )

        root = self._acquisition_root(frozen) / f"image-{execution.plan.image_id}"
        publish_acquisition_group(
            output_root=root,
            execution=execution,
            replayed=replayed,
            replay_receipt=parity,
        )
        return load_published_acquisition(root, plan=execution.plan)

    def _acquisition_root(self, frozen: Any) -> Path:
        del frozen
        root = self._acquisition_root_path
        if root is None:
            raise LiveCompositionError(
                "the acquisition root is only bound inside one sealed node"
            )
        return root

    def _manifest(self, frozen: Any) -> Any:
        from scripts.research.build_human13_k_union_manifest import load_manifest

        return load_manifest(frozen.manifest_path, require_full_panel=True)

    def _credit_ledger(self, frozen: Any, acquisition: Any) -> Any:
        from scripts.research.human13_trajectory_credit import (
            build_trajectory_credit_ledger,
        )

        manifest = self._manifest(frozen)
        return build_trajectory_credit_ledger(
            manifest,
            acquisition,
            tokenizer_adapter=self._backend.tokenizer_adapter(
                manifest=manifest,
                publication=acquisition.publications[0],
            ),
        )

    def _acquisition_evidence(
        self,
        *,
        frozen: Any,
        training_rp: float,
        acquisition: Any,
        credit_ledger: Any,
        compiler_ledger: Any,
        nested: Any,
        executions: Sequence[Any],
    ) -> AcquisitionEvidence:
        publications = acquisition.publications
        native_sha256s = tuple(
            item.binding.native_receipts_sha256 for item in publications
        )
        seed_groups = {item.plan.seed_group_id for item in executions}
        seed_rows = {
            tuple(request.seed for request in item.plan.requests) for item in executions
        }
        if len(seed_groups) != 1 or len(seed_rows) != 1:
            raise LiveCompositionError(
                "acquisition executions do not share one sealed seed group"
            )
        seed_group_id = seed_groups.pop()
        seeds = seed_rows.pop()
        shared = SharedEvidenceRef(
            source_sha256=frozen.source_checkpoint_payload_sha256,
            manifest_sha256=frozen.manifest_sha256,
            acquisition_path=str(self._acquisition_root(frozen)),
            acquisition_sha256=acquisition.content_sha256,
            trajectory_credit_acquisition_sha256=credit_ledger.acquisition_sha256,
            credit_ledger_sha256=credit_ledger.content_sha256,
            compiler_ledger_sha256=compiler_ledger.content_sha256,
            policy_contract_sha256=(
                publications[0].replayed_group.policy_contract.content_sha256
            ),
            native_receipts_sha256=_sha256(
                {
                    "schema_version": "human13_rp_crossover_native_bundle.v1",
                    "native_receipts_sha256s": list(native_sha256s),
                }
            ),
            training_rp=training_rp,
            seed_group_id=seed_group_id,
            seeds=seeds,
        )
        return AcquisitionEvidence(
            acquisition=acquisition,
            credit_ledger=credit_ledger,
            compiler_ledger=compiler_ledger,
            nested=nested,
            shared_evidence=shared,
            acquisition_path=shared.acquisition_path,
            native_receipts_sha256=shared.native_receipts_sha256,
            request_count=sum(len(item.plan.requests) for item in executions),
            batch_count=sum(len(item.plan.batches) for item in executions),
            token_count=sum(item.binding.token_count for item in publications),
        )

    def _cell_specs(
        self, node: Mapping[str, Any], evidence: AcquisitionEvidence
    ) -> tuple[CellSpec, ...]:
        acquisition_key = AcquisitionKey.from_dict(node["acquisition_key"])
        specs = []
        for planned in node["cells"]:
            cell_key = CellKey.from_dict(planned["cell_key"])
            if cell_key.acquisition_key != acquisition_key:
                raise LiveCompositionError(
                    "planned cell key differs from its acquisition node"
                )
            arm_id = cell_key.arm_id
            specs.append(
                CellSpec(
                    cell_key=cell_key,
                    shared_evidence=evidence.shared_evidence,
                    leaf_config_sha256=planned["source_leaf_config_sha256"],
                    source_checkpoint_sha256=evidence.shared_evidence.source_sha256,
                    expected_objective_components=(
                        (*PROPOSAL_COMPONENTS_BY_ARM[arm_id], "preservation")
                        if arm_id == "C"
                        else PROPOSAL_COMPONENTS_BY_ARM[arm_id]
                    ),
                    objective_component_hashes=tuple(
                        evidence.nested.arm_component_hashes(arm_id)
                    ),
                    adamw_config_sha256=planned["adamw_config_sha256"],
                    fresh_optimizer_identity_sha256=planned[
                        "fresh_optimizer_identity_sha256"
                    ],
                    evaluation_rps=EVALUATION_RPS,
                    output_root=planned["output_root"],
                    learning_rate=planned["learning_rate"],
                    global_learning_rate_decision_sha256=planned.get(
                        "global_learning_rate_decision_sha256"
                    ),
                    resolved_leaf_config_sha256=planned["resolved_leaf_config_sha256"],
                )
            )
        specs_tuple = tuple(specs)
        arms = tuple(spec.cell_key.arm_id for spec in specs_tuple)
        if acquisition_key.phase == "qualification":
            if len(specs_tuple) != 5 or arms != ("C",) * 5:
                raise LiveCompositionError(
                    "qualification requires exactly five independent C cells"
                )
        elif len(specs_tuple) != 3 or arms != ("A", "B", "C"):
            raise LiveCompositionError(
                "matrix acquisition requires exactly the nested A/B/C cells"
            )
        return specs_tuple

    # -- cell runtime services owner ----------------------------------------

    def services_for_cell(self, spec: CellSpec) -> Any:
        if self._evidence is None or self._witness_bank is None:
            raise LiveCompositionError(
                "cell services requested before the acquisition was materialized"
            )
        return _CellServices(composition=self, spec=spec)

    # -- witness/dose measurement -------------------------------------------

    def witness_bank(self, state: CellExecutionState) -> FrozenWitnessBank:
        frozen_bank = self._witness_bank
        source_flat = self._source_flat
        if frozen_bank is None or source_flat is None:
            raise LiveCompositionError("the frozen witness bank is unavailable")
        cell_flat = _flat_float64(state.named_trainable_parameters, frozen_bank.layout)
        if not torch.equal(cell_flat, source_flat):
            raise LiveCompositionError(
                "this cell's fresh Source differs from the frozen witness surface"
            )
        surface = getattr(state, "_human13_witness_surface", None)
        if surface is None:
            return frozen_bank
        measurement = witness_owner.WitnessMeasurement(
            decodes=self._sealed_decodes(), surface=surface
        )
        rebound = measurement.freeze_witness_bank(binding=frozen_bank.binding)
        if rebound.to_dict() != frozen_bank.to_dict():
            raise LiveCompositionError(
                "cell witness surface differs from the pre-acquisition frozen bank"
            )
        if measurement.teacher_forced_greedy_change_count() != 0:
            raise LiveCompositionError(
                "cell witness Source is not its own teacher-forced argmax"
            )
        object.__setattr__(state, "_human13_witness_measurement", measurement)
        object.__setattr__(state, "_human13_witness_bank", rebound)
        return rebound

    def realized_margin_probe(
        self, state: CellExecutionState, spec: CellSpec
    ) -> Callable[[], Mapping[str, float]]:
        measurement = getattr(state, "_human13_witness_measurement", None)
        surface = getattr(state, "_human13_witness_surface", None)
        bank = getattr(state, "_human13_witness_bank", None)
        if measurement is None:
            measurement = self._measurement
        if surface is None:
            surface = self._margin_surface
        if bank is None:
            bank = self._witness_bank
        source_flat = self._source_flat
        source_dose = self._source_dose_margins
        if (
            measurement is None
            or surface is None
            or bank is None
            or source_flat is None
            or source_dose is None
        ):
            raise LiveCompositionError("the frozen witness surface is unavailable")

        def probe() -> Mapping[str, float]:
            witness_owner.mirror_parameter_values(
                source=state.named_trainable_parameters,
                target=surface.named_trainable_parameters(),
            )
            try:
                realized = measurement.margin_values()
                applied_flat = _flat_float64(
                    surface.named_trainable_parameters(), bank.layout
                )
                applied_delta = applied_flat - source_flat
                jvp_error = witness_owner.jvp_finite_difference_error(
                    bank, applied_delta=applied_delta, realized=realized
                )
                source_median, displacement_median = (
                    witness_owner.decision_margin_dose_statistics(
                        source_margins=source_dose,
                        applied_margins=measurement.dose_site_margins(),
                    )
                )
                self._dose_mechanics[spec.cell_key.content_sha256] = {
                    "greedy_decision_change_count": (
                        measurement.teacher_forced_greedy_change_count()
                    ),
                    "jvp_fd_max_abs_error": jvp_error,
                    "jvp_fd_tolerance": witness_owner.JVP_FD_TOLERANCE,
                    "median_abs_source_decision_margin": source_median,
                    "median_abs_decision_margin_displacement": displacement_median,
                }
            finally:
                self._restore_margin_surface(surface=surface, bank=bank)
            return realized

        return probe

    def _restore_margin_surface(self, *, surface: Any, bank: FrozenWitnessBank) -> None:
        source_flat = self._source_flat
        if surface is None or source_flat is None:
            raise LiveCompositionError("the frozen witness surface is unavailable")
        offset = 0
        with torch.no_grad():
            values = dict(surface.named_trainable_parameters())
            for entry in bank.layout.entries:
                chunk = source_flat[offset : offset + entry.numel]
                offset += entry.numel
                parameter = values[entry.name]
                parameter.copy_(chunk.reshape(entry.shape).to(dtype=parameter.dtype))
        if offset != int(source_flat.numel()):
            raise LiveCompositionError("witness surface restore covered fewer entries")

    def dose_mechanics(
        self,
        spec: CellSpec,
        *,
        checkpoint: PrivateCheckpointRef,
        proposal_sha256: str,
        audits: Sequence[AuditRef],
        resources: AggregateResourceReceipt,
        projection: Any,
        output_deltas: Mapping[str, int],
    ) -> DoseMechanicalReceipt:
        measured = self._dose_mechanics.get(spec.cell_key.content_sha256)
        if measured is None:
            raise LiveCompositionError(
                "dose mechanics were never measured on the frozen witness surface"
            )
        return DoseMechanicalReceipt(
            cell_key=spec.cell_key,
            proposal_sha256=proposal_sha256,
            private_checkpoint_sha256=checkpoint.checkpoint_sha256,
            audit_checkpoint_sha256s=cast(
                "tuple[str, str]",
                tuple(audit.evaluated_checkpoint_sha256 for audit in audits),
            ),
            greedy_decision_change_count=int(measured["greedy_decision_change_count"]),
            malformed_output_delta_count=int(output_deltas["malformed"]),
            cap_terminated_output_delta_count=int(output_deltas["cap_terminated"]),
            unparseable_output_delta_count=int(output_deltas["unparseable"]),
            active_witness_count=len(tuple(projection.active_witnesses)),
            jvp_fd_max_abs_error=float(measured["jvp_fd_max_abs_error"]),
            jvp_fd_tolerance=float(measured["jvp_fd_tolerance"]),
            median_abs_decision_margin_displacement=float(
                measured["median_abs_decision_margin_displacement"]
            ),
            median_abs_source_decision_margin=float(
                measured["median_abs_source_decision_margin"]
            ),
            rollback_reproduced=True,
            resources=resources,
        )


@dataclass
class _CellServices:
    """One independent cell's runtime services; never reused across cells."""

    composition: Human13RPCrossoverLiveComposition
    spec: CellSpec

    def __post_init__(self) -> None:
        self._stack = ExitStack()
        self._started = time.perf_counter()
        self._backward: ObjectiveBackwardReceipt | None = None
        self._outcomes: list[AuditOutcome] = []
        self._state: CellExecutionState | None = None

    def open_cell(self, spec: CellSpec) -> CellExecutionState:
        if spec is not self.spec and spec != self.spec:
            raise LiveCompositionError("cell services received a different CellSpec")
        state = self.composition._backend.open_cell(spec, self.composition._frozen)
        self._state = state
        return state

    def backward_objective(
        self, state: CellExecutionState, spec: CellSpec
    ) -> ObjectiveBackwardReceipt:
        evidence = self.composition._evidence
        if evidence is None:
            raise LiveCompositionError("cell backward has no acquisition evidence")
        receipt = self.composition._backend.backward_objective(
            state, spec, evidence, self.composition._packed
        )
        self._backward = receipt
        return receipt

    def witness_bank(
        self, state: CellExecutionState, spec: CellSpec
    ) -> FrozenWitnessBank:
        del spec
        return self.composition.witness_bank(state)

    def realized_margin_probe(
        self,
        state: CellExecutionState,
        spec: CellSpec,
        bank: FrozenWitnessBank,
    ) -> Callable[[], Mapping[str, float]]:
        del bank
        measured_probe = self.composition.realized_margin_probe(state, spec)

        def probe_and_release() -> Mapping[str, float]:
            try:
                return measured_probe()
            finally:
                surface = getattr(state, "_human13_witness_surface", None)
                if surface is not None:
                    self.composition._backend.close_margin_surface(surface)
                    object.__setattr__(state, "_human13_witness_surface", None)
                    object.__setattr__(state, "_human13_witness_measurement", None)
                    object.__setattr__(state, "_human13_witness_bank", None)

        return probe_and_release

    def write_private_checkpoint(
        self, state: CellExecutionState, spec: CellSpec
    ) -> PrivateCheckpointRef:
        return self.composition._backend.write_private_checkpoint(
            state, spec, self._stack
        )

    def audit_checkpoint(
        self,
        state: CellExecutionState,
        checkpoint: PrivateCheckpointRef,
        repetition_penalty: float,
    ) -> AuditRef:
        del state
        outcome = self.composition._backend.audit_checkpoint(
            self.spec, checkpoint, repetition_penalty
        )
        if not isinstance(outcome, AuditOutcome):
            raise LiveCompositionError("audit backend must return a typed AuditOutcome")
        self._outcomes.append(outcome)
        return outcome.audit

    def cleanup_private_checkpoint(self, checkpoint: PrivateCheckpointRef) -> None:
        del checkpoint
        self._stack.close()

    def close_cell(self, state: CellExecutionState, spec: CellSpec) -> None:
        if spec is not self.spec and spec != self.spec:
            raise LiveCompositionError("cell close received a different CellSpec")
        try:
            self._stack.close()
        finally:
            self.composition._backend.close_cell(state)

    def aggregate_resource_receipt(
        self,
        state: CellExecutionState,
        spec: CellSpec,
        backward: ObjectiveBackwardReceipt,
        audits: Sequence[AuditRef],
    ) -> AggregateResourceReceipt:
        del state, spec
        evidence = self.composition._evidence
        if evidence is None:
            raise LiveCompositionError("resource receipt has no acquisition evidence")
        snapshot = self.composition._backend.resource_snapshot()
        if not isinstance(snapshot, ResourceSnapshot):
            raise LiveCompositionError("resource backend must return ResourceSnapshot")
        return AggregateResourceReceipt(
            measurement_scope=snapshot.measurement_scope,
            wall_time_seconds=time.perf_counter() - self._started,
            peak_host_rss_bytes=snapshot.peak_host_rss_bytes,
            cuda_peak_allocated_bytes=snapshot.cuda_peak_allocated_bytes,
            cuda_peak_reserved_bytes=snapshot.cuda_peak_reserved_bytes,
            acquisition_request_count=evidence.request_count,
            acquisition_batch_count=evidence.batch_count,
            acquisition_token_count=evidence.token_count,
            decode_request_count=len(audits) * len(CANONICAL_IMAGE_IDS),
            decode_batch_count=len(audits) * len(CANONICAL_IMAGE_IDS),
            decode_token_count=snapshot.decode_token_count,
            packed_token_count=snapshot.packed_token_count,
            logical_token_count=snapshot.logical_token_count,
            forward_count=snapshot.forward_count,
            backward_count=backward.backward_count,
            row_bytes=snapshot.row_bytes,
            artifact_bytes=snapshot.artifact_bytes,
            update_count=1,
            audit_count=len(audits),
            rollback_count=1,
        )

    def dose_mechanical_receipt(
        self,
        state: CellExecutionState,
        spec: CellSpec,
        checkpoint: PrivateCheckpointRef,
        proposal_sha256: str,
        audits: Sequence[AuditRef],
        resources: AggregateResourceReceipt,
        projection: Any,
    ) -> DoseMechanicalReceipt:
        del state
        return self.composition.dose_mechanics(
            spec,
            checkpoint=checkpoint,
            proposal_sha256=proposal_sha256,
            audits=audits,
            resources=resources,
            projection=projection,
            output_deltas={
                "malformed": sum(item.malformed_delta for item in self._outcomes),
                "cap_terminated": sum(
                    item.cap_terminated_delta for item in self._outcomes
                ),
                "unparseable": sum(item.unparseable_delta for item in self._outcomes),
            },
        )


__all__ = [
    "AcquisitionEvidence",
    "AuditOutcome",
    "Human13RPCrossoverLiveComposition",
    "LEGACY_M_OWNER_CLASS",
    "LiveCompositionBackend",
    "LiveCompositionError",
    "NATIVE_BATCH_COUNT",
    "NATIVE_REQUEST_COUNT",
    "QUALIFICATION_SEED_GROUP",
    "ResourceSnapshot",
    "SCHEMA_VERSION",
    "STREAMING_MODE",
    "SourceSurfaceEvidence",
    "TRUSTED_OWNER_CLASS",
    "UNIT_ID",
]
