"""Explicit live-owner seam for the all-HF image-1584 Task-5 vertical.

This module owns lifecycle and lineage only.  Canonical trajectory projection,
compiler construction, witness measurement, and proposal math remain with their
existing owners.  In particular, this seam never converts an HF group into the
older vLLM ``AdmittedPublication`` contract.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable
from weakref import ReferenceType, ref

from scripts.research.human13_hf_shared_surface import (
    GradientReplayGroup,
    SampledHFGroup,
)
from scripts.research.human13_hf_native_projection import (
    HFNativeProjectionError,
    hf_native_request_evidence_sha256 as _projected_request_evidence_sha256,
)
from scripts.research.human13_source_surface_reconciliation import (
    SOURCE_SURFACE,
    TRAINING_SURFACE,
    CanonicalSourceBaselineReceipt,
    CoordinateAliasReconciliation,
    SourceSurfaceReconciliationReceipt,
    SourceSurfaceReconciliationRequest,
    reconcile_coordinate_alias,
    reconcile_source_surface,
)
from src.artifacts.json_values import json_sha256
from src.data.geometry import (
    COORD_TOKEN_PATTERN,
    coord_bins_to_pixel_xyxy,
    parse_coord_token,
)

if TYPE_CHECKING:
    from scripts.research.human13_adamw_proposal_preservation import FrozenWitnessBank
    from scripts.research.human13_trajectory_credit import TrajectoryCreditLedger


class HFNativeOneImageOwnerError(RuntimeError):
    """Exact live owner evidence is absent, late, or lineage-incompatible."""

    def __init__(
        self,
        reason: str,
        *,
        disposition: str = "hf_native_owner_admission_failure",
        reconciliation_receipt: SourceSurfaceReconciliationReceipt | None = None,
    ) -> None:
        if not isinstance(reason, str) or not reason:
            raise TypeError("HF-native owner failure reason must be nonempty")
        if not isinstance(disposition, str) or not disposition:
            raise TypeError("HF-native owner failure disposition must be nonempty")
        self.reason = reason
        self.disposition = disposition
        self.reconciliation_receipt = reconciliation_receipt
        super().__init__(f"{disposition}: {reason}")


@dataclass(frozen=True)
class _CoordinateAliasDiagnosticContext:
    """One immutable snapshot used by every coordinate-alias failure path."""

    repetition_penalty: float | None = None
    source_tokens: tuple[str, ...] = ()
    training_tokens: tuple[str, ...] = ()
    source_token_ids: tuple[int, ...] = ()
    training_token_ids: tuple[int, ...] = ()
    coordinate_roles: tuple[tuple[int, str, str], ...] = ()
    source_boxes: tuple[tuple[str, tuple[float, ...]], ...] = ()
    training_boxes: tuple[tuple[str, tuple[float, ...]], ...] = ()
    source_owner_rows: tuple[tuple[str, int], ...] = ()
    training_owner_rows: tuple[tuple[str, int], ...] = ()
    source_membership: tuple[tuple[str, str], ...] = ()
    training_membership: tuple[tuple[str, str], ...] = ()
    source_protected_g: tuple[str, ...] = ()
    training_protected_g: tuple[str, ...] = ()

    @staticmethod
    def _boxes(
        values: Mapping[str, Sequence[float]],
    ) -> tuple[tuple[str, tuple[float, ...]], ...]:
        return tuple(
            (str(owner_id), tuple(float(value) for value in box))
            for owner_id, box in sorted(values.items())
        )

    @staticmethod
    def _rows(values: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
        return tuple(
            (str(owner_id), int(row)) for owner_id, row in sorted(values.items())
        )

    @staticmethod
    def _membership(values: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(
            (str(owner_id), str(member)) for owner_id, member in sorted(values.items())
        )

    @staticmethod
    def _roles(
        values: Mapping[int, tuple[str, str]],
    ) -> tuple[tuple[int, str, str], ...]:
        return tuple(
            (int(position), str(owner_id), str(role))
            for position, (owner_id, role) in sorted(values.items())
        )

    def to_kwargs(self) -> dict[str, object]:
        return {
            "repetition_penalty": self.repetition_penalty,
            "source_tokens": self.source_tokens,
            "training_tokens": self.training_tokens,
            "source_token_ids": self.source_token_ids,
            "training_token_ids": self.training_token_ids,
            "coordinate_roles": {
                position: (owner_id, role)
                for position, owner_id, role in self.coordinate_roles
            },
            "source_boxes": dict(self.source_boxes),
            "training_boxes": dict(self.training_boxes),
            "source_owner_rows": dict(self.source_owner_rows),
            "training_owner_rows": dict(self.training_owner_rows),
            "source_membership": dict(self.source_membership),
            "training_membership": dict(self.training_membership),
            "source_protected_g": self.source_protected_g,
            "training_protected_g": self.training_protected_g,
        }


def _project_coordinate_bbox(
    labels: Sequence[str],
    positions: Sequence[int],
    *,
    image_width: int,
    image_height: int,
    field: str,
) -> tuple[int, int, int, int]:
    """Project canonical norm-1000 coordinate tokens into pixel ``xyxy``."""

    if len(positions) != 4:
        raise ValueError(f"{field} must contain four coordinate positions")
    bins = tuple(
        parse_coord_token(labels[position], field=f"{field}[{index}]")
        for index, position in enumerate(positions)
    )
    return coord_bins_to_pixel_xyxy(
        bins,
        image_width=image_width,
        image_height=image_height,
        field=field,
    )


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise HFNativeOneImageOwnerError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise HFNativeOneImageOwnerError(f"{field} must be a SHA-256 digest") from error
    return value


def freeze_witness_bank_for_post_acquisition(
    witness_bank: object,
) -> FrozenWitnessBank:
    """Detach and digest-check every Source Jacobian before K16 acquisition."""

    from scripts.research.human13_adamw_proposal_preservation import (
        FrozenWitnessBank,
    )

    if type(witness_bank) is not FrozenWitnessBank:
        raise HFNativeOneImageOwnerError(
            "Source witness owner did not return a frozen witness bank",
            disposition="source_witness_bank_untyped",
        )
    bank = cast("FrozenWitnessBank", witness_bank)
    jacobians = {
        witness.canonical_key: jacobian.detach().to(device="cpu").clone()
        for _index, witness, jacobian in bank.stream_constraints()
    }
    return FrozenWitnessBank.from_witnesses(
        (*bank.constraints, *bank.audit_only),
        jacobians=jacobians,
        binding=bank.binding,
        layout=bank.layout,
    )


@dataclass(frozen=True, init=False)
class AdmittedPostApplyMarginProbe:
    """Weak session bridge admitted only on a non-Source parameter state."""

    _session_ref: ReferenceType[object]
    source_parameter_state_sha256: str
    source_decodes: tuple[object, ...]
    witness_sites: tuple[object, ...]

    def __init__(
        self,
        *,
        session: object,
        source_parameter_state_sha256: str,
        source_decodes: tuple[object, ...],
        witness_sites: tuple[object, ...],
    ) -> None:
        try:
            session_ref = ref(session)
        except TypeError as error:
            raise HFNativeOneImageOwnerError(
                "post-apply margin session must support weak references",
                disposition="post_apply_session_not_weakrefable",
            ) from error
        object.__setattr__(self, "_session_ref", session_ref)
        object.__setattr__(
            self,
            "source_parameter_state_sha256",
            _digest(
                source_parameter_state_sha256,
                field="source_parameter_state_sha256",
            ),
        )
        object.__setattr__(self, "source_decodes", tuple(source_decodes))
        sites = tuple(witness_sites)
        if not sites:
            raise HFNativeOneImageOwnerError(
                "post-apply margin probe requires frozen constraint sites",
                disposition="post_apply_witness_sites_empty",
            )
        object.__setattr__(self, "witness_sites", sites)

    @property
    def content_sha256(self) -> str:
        return json_sha256(
            {
                "schema_version": "human13_hf_native_post_apply_margin_probe.v1",
                "source_parameter_state_sha256": self.source_parameter_state_sha256,
                "source_decode_keys": [
                    getattr(decode, "surface_key", None)
                    for decode in self.source_decodes
                ],
                "witness_site_keys": [
                    getattr(site, "canonical_key", None) for site in self.witness_sites
                ],
            }
        )

    def __call__(self) -> Mapping[str, float]:
        from scripts.research.human13_adamw_proposal_preservation import (
            ParameterLayout,
            parameter_state_sha256,
        )

        session = self._session_ref()
        if session is None:
            raise HFNativeOneImageOwnerError(
                "post-apply margin session was released before measurement",
                disposition="post_apply_session_released",
            )
        named_provider = getattr(session, "named_trainable_parameters", None)
        measure = getattr(session, "proposal_realized_margin_values", None)
        if not callable(named_provider) or not callable(measure):
            raise HFNativeOneImageOwnerError(
                "shared session lacks the admitted post-apply margin surface",
                disposition="post_apply_margin_surface_unavailable",
            )
        named = tuple(
            cast(
                Callable[..., tuple[tuple[str, Any], ...]],
                named_provider,
            )(allow_parameter_update=True)
        )
        layout = ParameterLayout.from_named_parameters(named)
        proposal_sha256 = parameter_state_sha256(named, layout)
        measured = measure(
            self.source_decodes,
            self.witness_sites,
            source_parameter_state_sha256=self.source_parameter_state_sha256,
            proposal_parameter_state_sha256=proposal_sha256,
        )
        if not isinstance(measured, Mapping):
            raise HFNativeOneImageOwnerError(
                "post-apply margin surface returned an untyped mapping",
                disposition="post_apply_margin_result_untyped",
            )
        result = {str(key): float(value) for key, value in measured.items()}
        expected = {str(getattr(site, "canonical_key")) for site in self.witness_sites}
        if set(result) != expected or any(
            not math.isfinite(value) for value in result.values()
        ):
            raise HFNativeOneImageOwnerError(
                "post-apply margins differ from frozen constraint coverage",
                disposition="post_apply_margin_coverage_mismatch",
            )
        return result


@dataclass(frozen=True)
class _HFNativeEvidenceIdentity:
    request_id: str
    generated_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class _HFNativePolicyContract:
    content_sha256: str


@dataclass(frozen=True)
class _HFNativeCreditEvidence:
    identity: _HFNativeEvidenceIdentity
    generated_tokens: tuple[int, ...]
    terminal_kind: str
    content_sha256: str
    policy_contract: _HFNativePolicyContract


def hf_native_request_evidence_sha256(
    sampled_request: object,
    replay_group: GradientReplayGroup,
) -> str:
    """Content identity used by native canonical projections and credit rows."""

    try:
        return _projected_request_evidence_sha256(
            cast(Any, sampled_request), replay_group
        )
    except HFNativeProjectionError as error:
        raise HFNativeOneImageOwnerError(str(error)) from error


def construct_hf_native_one_image_trajectory_ledger(
    *,
    manifest: object,
    manifest_image: object,
    replay_groups: tuple[GradientReplayGroup, ...],
    canonical_projections: tuple[object, ...],
    source_checkpoint_sha256: str,
) -> TrajectoryCreditLedger:
    """Build admitted K16 credit from HF replay and canonical parser spans.

    This is deliberately separate from the legacy Task-2 publication builder.
    It consumes no vLLM ``AdmittedPublication`` and reuses only the canonical
    parser projection value type and the frozen credit projection math.
    """

    from scripts.research import human13_trajectory_credit as credit
    from scripts.research.build_human13_k_union_manifest import (
        Human13KUnionManifest,
    )

    _digest(source_checkpoint_sha256, field="source_checkpoint_sha256")
    if not isinstance(manifest, Human13KUnionManifest):
        raise HFNativeOneImageOwnerError(
            "native trajectory requires canonical manifest"
        )
    if getattr(manifest_image, "image_id", None) != 1584:
        raise HFNativeOneImageOwnerError("native trajectory image differs from 1584")
    if (
        tuple(getattr(image, "image_id", None) for image in manifest.images).count(1584)
        != 1
    ):
        raise HFNativeOneImageOwnerError(
            "native trajectory manifest image is ambiguous"
        )
    groups = tuple(replay_groups)
    if (
        len(groups) != 4
        or any(type(group) is not GradientReplayGroup for group in groups)
        or tuple(group.sampled_group.group_index for group in groups) != (0, 1, 2, 3)
    ):
        raise HFNativeOneImageOwnerError("native trajectory requires four HF replays")
    identities = {group.sampled_group.identity for group in groups}
    if len(identities) != 1:
        raise HFNativeOneImageOwnerError("native trajectory shared surface differs")
    sampled_requests = tuple(
        request for group in groups for request in group.sampled_group.requests
    )
    projections = tuple(canonical_projections)
    projections_live = cast(tuple[Any, ...], projections)
    if (
        len(sampled_requests) != 16
        or len(projections) != 16
        or any(
            type(projection) is not credit.CanonicalTrajectoryProjection
            for projection in projections
        )
        or tuple(projection.request_id for projection in projections_live)
        != tuple(request.request_id for request in sampled_requests)
    ):
        raise HFNativeOneImageOwnerError(
            "native trajectory requires sixteen canonical parser projections"
        )
    policy_sha256s = tuple(
        json_sha256(group.sampled_group.policy.to_dict()) for group in groups
    )
    evidence: list[_HFNativeCreditEvidence] = []
    parsed: list[Any] = []
    projection_index = 0
    for group, policy_sha256 in zip(groups, policy_sha256s, strict=True):
        for request in group.sampled_group.requests:
            projection = projections_live[projection_index]
            generated = tuple(token.chosen_token_id for token in request.tokens)
            evidence_sha256 = hf_native_request_evidence_sha256(request, group)
            if projection.acquisition_trajectory_sha256 != evidence_sha256:
                raise HFNativeOneImageOwnerError(
                    "canonical parser projection differs from HF replay evidence"
                )
            evidence.append(
                _HFNativeCreditEvidence(
                    identity=_HFNativeEvidenceIdentity(request.request_id, generated),
                    generated_tokens=generated,
                    terminal_kind=(
                        "natural_stop" if request.stop_reason == "im_end" else "cap"
                    ),
                    content_sha256=evidence_sha256,
                    policy_contract=_HFNativePolicyContract(policy_sha256),
                )
            )
            parsed.append(
                credit._ParsedTrajectoryCreditInput(
                    request_id=request.request_id,
                    rows=tuple(
                        credit._ParsedCreditRow(
                            generated_order=event.event_order,
                            category=event.category,
                            bbox=event.bbox,
                            token_start=event.token_start,
                            token_end=event.token_end,
                            geometry_valid=event.kind == "prediction",
                        )
                        for event in projection.events
                        if event.kind in {"prediction", "invalid"}
                    ),
                    malformed_spans=tuple(
                        credit._MalformedRowSpan(
                            generated_order=event.event_order,
                            token_start=event.token_start,
                            token_end=event.token_end,
                        )
                        for event in projection.events
                        if event.kind == "malformed"
                    ),
                )
            )
            projection_index += 1

    matcher = manifest.binding.matcher
    owners = tuple(getattr(manifest_image, "owners", ()))
    trusted = tuple(owner.owner_id for owner in owners if owner.stratum in {"G", "H"})
    legacy = tuple(owner.owner_id for owner in owners if owner.stratum == "M")
    if not trusted:
        raise HFNativeOneImageOwnerError("native trajectory has no trusted owners")
    owner_weight = 1.0 / len(trusted)
    trajectories = tuple(
        credit._project_one_trajectory(
            cast(Any, manifest_image),
            item,
            projection,
            trusted_owner_ids=trusted,
            legacy_owner_ids=legacy,
            owner_weight=owner_weight,
            duplicate_iou_threshold=matcher.duplicate_iou_threshold,
            owner_iou_threshold=matcher.owner_iou_threshold,
        )
        for item, projection in zip(evidence, parsed, strict=True)
    )
    trajectories, position_returns = credit._attach_rloo(trajectories)
    sampled_hashes = tuple(group.sampled_group.content_sha256 for group in groups)
    replay_hashes = tuple(group.content_sha256 for group in groups)
    projection_sha256 = json_sha256(
        [projection.to_dict() for projection in projections_live]
    )
    acquisition_sha256 = json_sha256(
        {
            "schema_version": "human13_hf_native_trajectory_acquisition.v1",
            "image_id": 1584,
            "sampled_group_sha256s": list(sampled_hashes),
            "replay_group_sha256s": list(replay_hashes),
            "parser_projection_sha256": projection_sha256,
        }
    )
    image_ledger = credit.ImageCreditLedger(
        image_id=1584,
        acquisition_group_sha256=acquisition_sha256,
        trusted_owner_ids=trusted,
        legacy_m_owner_ids=legacy,
        owner_weight=owner_weight,
        trajectories=trajectories,
        position_returns=position_returns,
        plan_sha256=json_sha256(groups[0].sampled_group.plan.to_dict()),
        native_receipts_sha256=json_sha256(list(sampled_hashes)),
        parity_receipt_sha256=json_sha256(
            [group.parity.content_sha256 for group in groups]
        ),
        parser_projection_sha256=projection_sha256,
    )
    ledger = credit._construct_trajectory_credit_ledger(
        source_sha256=source_checkpoint_sha256,
        manifest_sha256=credit._manifest_sha256(manifest),
        acquisition_sha256=acquisition_sha256,
        logical_image_count=1,
        logical_k=16,
        images=(image_ledger,),
        training_repetition_penalty=1.0,
        seed_group_id="35001..35016",
        admit_scientific=True,
    )
    return credit._require_scientific_ledger_admission(ledger)


@dataclass(frozen=True)
class SourceOwnerRequest:
    assembly: object
    session: object
    manifest: object
    manifest_image: object
    config: object
    source_audits: Mapping[float, Mapping[str, Any]]
    training_source_audits: Mapping[float, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        if tuple(self.source_audits) != (1.0, 1.1):
            raise HFNativeOneImageOwnerError(
                "source owner requires ordered RP1.0/RP1.1 audits"
            )
        if getattr(self.manifest_image, "image_id", None) != 1584:
            raise HFNativeOneImageOwnerError("source owner image differs from 1584")
        if getattr(self.assembly, "model", None) is None:
            raise HFNativeOneImageOwnerError("source owner lacks the live model")


@dataclass(frozen=True)
class PreAcquisitionSourceOwners:
    session_object_id: int
    model_object_id: int
    parameter_state_sha256: str
    manifest_sha256: str
    image_sha256: str
    source_checkpoint_sha256: str
    source_audit_sha256s: tuple[tuple[float, str], ...]
    compiler_source_context: object
    witness_bank: object
    realized_margin_probe: Callable[[], Mapping[str, float]]
    frozen_before_acquisition: bool
    sample_group_count_at_freeze: int
    replay_group_count_at_freeze: int
    source_decodes: tuple[object, ...] = ()
    compiler_raw_logits: object | None = None
    surface_reconciliation: SourceSurfaceReconciliationReceipt | None = None

    def __post_init__(self) -> None:
        _digest(self.source_checkpoint_sha256, field="source_checkpoint_sha256")
        if self.frozen_before_acquisition is not True:
            raise HFNativeOneImageOwnerError(
                "witness owner was frozen after acquisition"
            )
        if self.sample_group_count_at_freeze or self.replay_group_count_at_freeze:
            raise HFNativeOneImageOwnerError(
                "source compiler/witness owner must precede every sample/replay group"
            )
        if tuple(rp for rp, _ in self.source_audit_sha256s) != (1.0, 1.1):
            raise HFNativeOneImageOwnerError("source owner audit lineage differs")
        if not callable(self.realized_margin_probe):
            raise HFNativeOneImageOwnerError("source owner lacks a realized probe")
        if self.compiler_raw_logits is not None:
            import torch

            raw_logits = self.compiler_raw_logits
            if (
                not isinstance(raw_logits, torch.Tensor)
                or raw_logits.ndim != 2
                or raw_logits.shape[0] != 1
                or not raw_logits.requires_grad
                or not bool(torch.isfinite(raw_logits.detach()).all().item())
            ):
                raise HFNativeOneImageOwnerError(
                    "pre-acquisition compiler row must be one finite graph tensor"
                )
        if self.surface_reconciliation is not None and not isinstance(
            self.surface_reconciliation, SourceSurfaceReconciliationReceipt
        ):
            raise HFNativeOneImageOwnerError(
                "Source surface reconciliation receipt is untyped"
            )
        # Cross-surface token/row/owner divergence is diagnostic evidence under
        # the successor semantics.  The outer receipt must still admit strict
        # identity and the BF16-native Source owner; a nested coordinate checker
        # may therefore be a non-admitted diagnostic result.

    @property
    def content_sha256(self) -> str:
        return json_sha256(
            {
                "schema_version": "human13_hf_native_pre_acquisition_owners.v1",
                "session_object_id": self.session_object_id,
                "model_object_id": self.model_object_id,
                "parameter_state_sha256": self.parameter_state_sha256,
                "manifest_sha256": self.manifest_sha256,
                "image_sha256": self.image_sha256,
                "source_checkpoint_sha256": self.source_checkpoint_sha256,
                "source_audit_sha256s": [
                    [rp, digest] for rp, digest in self.source_audit_sha256s
                ],
                "frozen_before_acquisition": self.frozen_before_acquisition,
                "sample_group_count_at_freeze": self.sample_group_count_at_freeze,
                "replay_group_count_at_freeze": self.replay_group_count_at_freeze,
                "source_decode_sha256s": [
                    json_sha256(
                        {
                            "image_id": getattr(decode, "image_id", None),
                            "repetition_penalty": getattr(
                                decode, "repetition_penalty", None
                            ),
                            "prompt_token_ids": list(
                                getattr(decode, "prompt_token_ids", ())
                            ),
                            "generated_token_ids": list(
                                getattr(decode, "generated_token_ids", ())
                            ),
                        }
                    )
                    for decode in self.source_decodes
                ],
                "compiler_source_boundary_sha256s": [
                    getattr(boundary, "source_decode_sha256", None)
                    for boundary in (
                        self.compiler_source_context
                        if isinstance(self.compiler_source_context, tuple)
                        else ()
                    )
                ],
                "compiler_raw_logits": None
                if self.compiler_raw_logits is None
                else {
                    "object_id": id(self.compiler_raw_logits),
                    "shape": list(getattr(self.compiler_raw_logits, "shape", ())),
                    "dtype": str(getattr(self.compiler_raw_logits, "dtype", None)),
                    "requires_grad": getattr(
                        self.compiler_raw_logits, "requires_grad", None
                    ),
                },
                "surface_reconciliation_sha256": None
                if self.surface_reconciliation is None
                else self.surface_reconciliation.content_sha256,
            }
        )


@dataclass(frozen=True)
class HFNativeAdmissionRequest:
    assembly: object
    session: object
    manifest: object
    manifest_image: object
    config: object
    sampled_groups: tuple[object, ...]
    replay_groups: tuple[object, ...]
    replay_logprob_tensors: Mapping[str, object]

    def __post_init__(self) -> None:
        if len(self.sampled_groups) != 4 or any(
            type(group) is not SampledHFGroup for group in self.sampled_groups
        ):
            raise HFNativeOneImageOwnerError(
                "HF-native admission rejects old/native-publication surrogates"
            )
        if len(self.replay_groups) != 4 or any(
            type(group) is not GradientReplayGroup for group in self.replay_groups
        ):
            raise HFNativeOneImageOwnerError(
                "HF-native admission requires four exact gradient replay groups"
            )
        sampled_live = cast(tuple[SampledHFGroup, ...], self.sampled_groups)
        replay_live = cast(tuple[GradientReplayGroup, ...], self.replay_groups)
        for sampled, replay in zip(sampled_live, replay_live, strict=True):
            if replay.sampled_group is not sampled and replay.sampled_group != sampled:
                raise HFNativeOneImageOwnerError(
                    "HF-native replay lineage differs from its sampled group"
                )
        if not self.replay_logprob_tensors:
            raise HFNativeOneImageOwnerError("HF-native admission lacks graph tensors")


@dataclass(frozen=True)
class HFNativeTrajectoryAdmission:
    """Canonical K16 credit bound directly to the four live HF replay groups."""

    source_owner_sha256: str
    session_object_id: int
    model_object_id: int
    manifest_object_id: int
    manifest_sha256: str
    image_id: int
    image_sha256: str
    source_checkpoint_sha256: str
    parameter_state_sha256: str
    sampled_group_sha256s: tuple[str, ...]
    replay_group_sha256s: tuple[str, ...]
    parser_projection_sha256: str
    trajectory_ledger: object

    def __post_init__(self) -> None:
        from scripts.research.human13_trajectory_credit import (
            _require_scientific_ledger_admission,
        )

        for field in (
            "source_owner_sha256",
            "manifest_sha256",
            "image_sha256",
            "source_checkpoint_sha256",
            "parameter_state_sha256",
            "parser_projection_sha256",
        ):
            _digest(getattr(self, field), field=field)
        for field in ("session_object_id", "model_object_id", "manifest_object_id"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise HFNativeOneImageOwnerError(f"{field} must be an object identity")
        if self.image_id != 1584:
            raise HFNativeOneImageOwnerError(
                "trajectory admission image differs from 1584"
            )
        if len(self.sampled_group_sha256s) != 4 or len(self.replay_group_sha256s) != 4:
            raise HFNativeOneImageOwnerError(
                "trajectory admission requires the same four sample/replay groups"
            )
        for label, values in (
            ("sampled group", self.sampled_group_sha256s),
            ("replay group", self.replay_group_sha256s),
        ):
            for value in values:
                _digest(value, field=label)
        try:
            ledger = _require_scientific_ledger_admission(self.trajectory_ledger)
        except ValueError as error:
            raise HFNativeOneImageOwnerError(
                "trajectory admission requires an actual scientific ledger"
            ) from error
        if (
            ledger.source_sha256 != self.source_checkpoint_sha256
            or ledger.manifest_sha256 != self.manifest_sha256
            or ledger.logical_image_count != 1
            or ledger.logical_k != 16
            or tuple(image.image_id for image in ledger.images) != (1584,)
            or ledger.images[0].parser_projection_sha256
            != self.parser_projection_sha256
        ):
            raise HFNativeOneImageOwnerError(
                "trajectory ledger differs from native parser/source lineage"
            )

    @property
    def content_sha256(self) -> str:
        ledger = cast(Any, self.trajectory_ledger)
        return json_sha256(
            {
                "schema_version": "human13_hf_native_trajectory_admission.v1",
                "source_owner_sha256": self.source_owner_sha256,
                "session_object_id": self.session_object_id,
                "model_object_id": self.model_object_id,
                "manifest_object_id": self.manifest_object_id,
                "manifest_sha256": self.manifest_sha256,
                "image_id": self.image_id,
                "image_sha256": self.image_sha256,
                "source_checkpoint_sha256": self.source_checkpoint_sha256,
                "parameter_state_sha256": self.parameter_state_sha256,
                "sampled_group_sha256s": list(self.sampled_group_sha256s),
                "replay_group_sha256s": list(self.replay_group_sha256s),
                "parser_projection_sha256": self.parser_projection_sha256,
                "trajectory_ledger_sha256": ledger.content_sha256,
                "trajectory_admission_sha256": ledger.admission_sha256,
            }
        )


@dataclass(frozen=True)
class HFNativeCompilerAdmission:
    """Same-session Source-boundary compiler and graph-bearing compact rows."""

    source_owner_sha256: str
    trajectory_admission_sha256: str
    session_object_id: int
    model_object_id: int
    manifest_object_id: int
    manifest_sha256: str
    image_id: int
    image_sha256: str
    source_checkpoint_sha256: str
    source_boundary_sha256s: tuple[str, ...]
    compiler_ledger: object
    compiler_compact_logits: object

    def __post_init__(self) -> None:
        from scripts.research.human13_greedy_compiler import (
            _require_compact_logits,
            _require_compiler_admission,
        )

        for field in (
            "source_owner_sha256",
            "trajectory_admission_sha256",
            "manifest_sha256",
            "image_sha256",
            "source_checkpoint_sha256",
        ):
            _digest(getattr(self, field), field=field)
        for field in ("session_object_id", "model_object_id", "manifest_object_id"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise HFNativeOneImageOwnerError(f"{field} must be an object identity")
        if self.image_id != 1584:
            raise HFNativeOneImageOwnerError(
                "compiler admission image differs from 1584"
            )
        if len(self.source_boundary_sha256s) != 1:
            raise HFNativeOneImageOwnerError(
                "one-image compiler requires one exact Source boundary"
            )
        for value in self.source_boundary_sha256s:
            _digest(value, field="source boundary")
        try:
            ledger = _require_compiler_admission(self.compiler_ledger)
            compact = _require_compact_logits(self.compiler_compact_logits, ledger)
        except ValueError as error:
            raise HFNativeOneImageOwnerError(
                "compiler admission requires actual ledger and compact graph evidence"
            ) from error
        if (
            ledger.source_sha256 != self.source_checkpoint_sha256
            or ledger.manifest_sha256 != self.manifest_sha256
            or ledger.logical_image_count != 1
            or tuple(image.image_id for image in ledger.images) != (1584,)
            or tuple(image.source_decode_sha256 for image in ledger.images)
            != self.source_boundary_sha256s
        ):
            raise HFNativeOneImageOwnerError(
                "compiler ledger differs from same-session Source boundary"
            )
        raw_logits = cast(Any, compact)._raw_logits
        if not raw_logits or any(
            not bool(getattr(tensor, "requires_grad", False))
            for tensor in raw_logits.values()
        ):
            raise HFNativeOneImageOwnerError(
                "compiler compact admission lacks graph-bearing raw logits"
            )

    @property
    def content_sha256(self) -> str:
        ledger = cast(Any, self.compiler_ledger)
        compact = cast(Any, self.compiler_compact_logits)
        return json_sha256(
            {
                "schema_version": "human13_hf_native_compiler_admission.v1",
                "source_owner_sha256": self.source_owner_sha256,
                "trajectory_admission_sha256": self.trajectory_admission_sha256,
                "session_object_id": self.session_object_id,
                "model_object_id": self.model_object_id,
                "manifest_object_id": self.manifest_object_id,
                "manifest_sha256": self.manifest_sha256,
                "image_id": self.image_id,
                "image_sha256": self.image_sha256,
                "source_checkpoint_sha256": self.source_checkpoint_sha256,
                "source_boundary_sha256s": list(self.source_boundary_sha256s),
                "compiler_ledger_sha256": ledger.content_sha256,
                "compiler_admission_sha256": ledger.admission_sha256,
                "compact_admission_sha256": compact.admission_sha256,
                "compact_tensor_object_ids": [
                    [site_id, id(compact._raw_logits[site_id])]
                    for site_id in compact.site_ids
                ],
            }
        )


@dataclass(frozen=True)
class HFNativeOneImageAdmission:
    """Exact join of current HF groups to canonical admitted runtime evidence."""

    source_owner_sha256: str
    session_object_id: int
    model_object_id: int
    sampled_group_sha256s: tuple[str, ...]
    replay_group_sha256s: tuple[str, ...]
    runtime_evidence: object

    def __post_init__(self) -> None:
        if len(self.sampled_group_sha256s) != 4 or len(self.replay_group_sha256s) != 4:
            raise HFNativeOneImageOwnerError(
                "HF-native admission requires exact 4+4 hashes"
            )
        if self.runtime_evidence is None:
            raise HFNativeOneImageOwnerError(
                "HF-native admission lacks runtime evidence"
            )


@runtime_checkable
class HFNativeOneImageOwner(Protocol):
    def prepare_source(
        self, request: SourceOwnerRequest
    ) -> PreAcquisitionSourceOwners: ...

    def admit_after_replay(
        self,
        request: HFNativeAdmissionRequest,
        source: PreAcquisitionSourceOwners,
    ) -> HFNativeOneImageAdmission: ...


class RepositoryHFNativeOneImageOwner:
    """Public composition hook for repository-owned canonical live owners.

    The two callables are deliberately explicit construction dependencies, not
    hidden attributes on ``HFSharedSurfaceSession``.  A production factory must
    supply the real canonical source/witness and HF trajectory/compiler owners;
    tests may supply inert value owners.
    """

    def __init__(
        self,
        *,
        prepare_source_owner: Callable[
            [SourceOwnerRequest], PreAcquisitionSourceOwners
        ],
        admit_hf_owners: Callable[
            [HFNativeAdmissionRequest, PreAcquisitionSourceOwners],
            HFNativeOneImageAdmission,
        ],
    ) -> None:
        if not callable(prepare_source_owner) or not callable(admit_hf_owners):
            raise TypeError("HF-native owner callbacks must be callable")
        self._prepare = prepare_source_owner
        self._admit = admit_hf_owners

    def prepare_source(self, request: SourceOwnerRequest) -> PreAcquisitionSourceOwners:
        result = self._prepare(request)
        if type(result) is not PreAcquisitionSourceOwners:
            raise HFNativeOneImageOwnerError(
                "source owner returned an untyped pre-acquisition context"
            )
        if result.session_object_id != id(
            request.session
        ) or result.model_object_id != id(getattr(request.assembly, "model")):
            raise HFNativeOneImageOwnerError("source owner substituted live objects")
        return result

    def admit_after_replay(
        self,
        request: HFNativeAdmissionRequest,
        source: PreAcquisitionSourceOwners,
    ) -> HFNativeOneImageAdmission:
        if source.session_object_id != id(request.session):
            raise HFNativeOneImageOwnerError(
                "acquisition session differs from Source owner"
            )
        result = self._admit(request, source)
        if type(result) is not HFNativeOneImageAdmission:
            raise HFNativeOneImageOwnerError(
                "HF-native owner returned an untyped admission"
            )
        sampled_live = cast(tuple[SampledHFGroup, ...], request.sampled_groups)
        replay_live = cast(tuple[GradientReplayGroup, ...], request.replay_groups)
        sampled_hashes = tuple(group.content_sha256 for group in sampled_live)
        replay_hashes = tuple(group.content_sha256 for group in replay_live)
        if (
            result.source_owner_sha256 != source.content_sha256
            or result.session_object_id != id(request.session)
            or result.model_object_id != id(getattr(request.assembly, "model"))
            or result.sampled_group_sha256s != sampled_hashes
            or result.replay_group_sha256s != replay_hashes
        ):
            raise HFNativeOneImageOwnerError("HF-native admission changed live lineage")
        return result


def prepare_repository_source_owners(
    request: SourceOwnerRequest,
) -> PreAcquisitionSourceOwners:
    """Project real Source audits and freeze witnesses on the same HF session."""

    from scripts.research.analyze_human13_k_union import _match_prefix
    from scripts.research.build_human13_on_policy_frontier import (
        CheckpointIdentity,
        CurrentDecode,
        CurrentPrediction,
        _project_image,
    )
    from scripts.research.human13_adamw_proposal_preservation import (
        LEGACY_M_OWNER_CLASS,
        TRUSTED_OWNER_CLASS,
        WitnessBinding,
    )
    from scripts.research.human13_all_hf_vertical import UNIT_ID
    from scripts.research.human13_greedy_compiler import (
        SourceBoundaryInput,
        _build_image_ledger,
        _frozen_alias_bank,
    )
    from scripts.research.human13_rp_crossover_witness import (
        SealedOwnerRow,
        SealedSourceDecode,
        WitnessMeasurement,
    )
    from src.inference.parsing import parse_compact_object_box_closed
    from src.templates.renderer import (
        BOX_END_TOKEN,
        OBJECT_REF_END_TOKEN,
        OBJECT_REF_START_TOKEN,
    )

    identity = getattr(request.session, "source_owner_identity", None)
    if identity is None or not callable(
        getattr(request.session, "raw_logit_rows", None)
    ):
        raise HFNativeOneImageOwnerError(
            "shared session lacks the public Source raw-logit owner surface"
        )
    matcher = getattr(getattr(request.manifest, "binding", None), "matcher", None)
    if matcher is None:
        raise HFNativeOneImageOwnerError("manifest lacks canonical owner matcher")
    sealed: list[SealedSourceDecode] = []
    boundaries: list[SourceBoundaryInput] = []
    audit_hashes: list[tuple[float, str]] = []
    training_audit_hashes: list[tuple[float, str]] = []
    checkpoint_hashes: set[str] = set()
    source_runtime_identities: list[Mapping[str, object]] = []
    source_checkpoint_payload_sha256s: list[str] = []
    source_checkpoint_paths: list[str] = []
    source_tokenizer_sha256s: list[str] = []
    source_prompt_sha256s: list[str] = []
    source_image_sha256s: list[str] = []
    source_manifest_sha256s: list[str] = []
    source_adapter_sha256s: list[str] = []
    source_embedding_delta_sha256s: list[str] = []
    source_base_model_paths: list[str] = []
    source_surface_snapshots: list[
        tuple[
            float,
            tuple[int, ...],
            tuple[Mapping[str, Any], ...],
            Mapping[str, Any],
        ]
    ] = []
    training_surface_snapshots: list[
        tuple[
            float,
            tuple[int, ...],
            tuple[Mapping[str, Any], ...],
            Mapping[str, Any],
        ]
    ] = []
    canonical_baselines: list[CanonicalSourceBaselineReceipt] = []
    source_image_dimensions: tuple[int, int] | None = None
    manifest_image = cast(Any, request.manifest_image)
    surface = cast(Any, request.session)
    owners = {owner.owner_id: owner for owner in manifest_image.owners}

    def _token_labels(token_ids: Sequence[int]) -> tuple[str, ...]:
        tokenizer = getattr(surface, "_tokenizer", None)
        convert = getattr(tokenizer, "convert_ids_to_tokens", None)
        if not callable(convert):
            raise HFNativeOneImageOwnerError(
                "shared session lacks canonical token-string conversion"
            )
        labels: list[str] = []
        for token_id in token_ids:
            value = convert(int(token_id))
            if isinstance(value, list):
                if len(value) != 1:
                    raise HFNativeOneImageOwnerError(
                        "canonical tokenizer returned multiple token strings"
                    )
                value = value[0]
            labels.append(str(value))
        return tuple(labels)

    def _row_token_spans(labels: Sequence[str]) -> tuple[tuple[int, int], ...]:
        spans: list[tuple[int, int]] = []
        index = 0
        while index < len(labels):
            if labels[index] != OBJECT_REF_START_TOKEN:
                index += 1
                continue
            object_end = next(
                (
                    position
                    for position in range(index + 1, len(labels))
                    if labels[position] == OBJECT_REF_END_TOKEN
                ),
                None,
            )
            if object_end is None:
                break
            box_end = next(
                (
                    position
                    for position in range(object_end + 1, len(labels))
                    if labels[position] == BOX_END_TOKEN
                ),
                None,
            )
            if box_end is None:
                break
            spans.append((index, box_end + 1))
            index = box_end + 1
        return tuple(spans)

    def _build_bf16_source_audits() -> dict[float, Mapping[str, Any]]:
        provided = request.training_source_audits
        if provided is not None:
            if tuple(provided) != (1.0, 1.1):
                raise HFNativeOneImageOwnerError(
                    "BF16 Source audits must be ordered RP1.0/RP1.1"
                )
            return dict(provided)
        free_running = getattr(surface, "free_running_greedy_token_ids", None)
        tokenizer = getattr(surface, "_tokenizer", None)
        decode_text = getattr(tokenizer, "decode", None)
        if not callable(free_running) or not callable(decode_text):
            raise HFNativeOneImageOwnerError(
                "shared session lacks BF16-native free-running Source projection"
            )
        dimensions = source_image_dimensions
        if dimensions is None:
            raise HFNativeOneImageOwnerError(
                "fp32 Source audit lacks image dimensions for BF16 projection"
            )
        width, height = dimensions
        result: dict[float, Mapping[str, Any]] = {}
        for rp in (1.0, 1.1):
            generated = tuple(
                int(value)
                for value in cast(Sequence[Any], free_running(repetition_penalty=rp))
            )
            labels = _token_labels(generated)
            text = str(
                decode_text(
                    list(generated),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            parsed = parse_compact_object_box_closed(
                text,
                row_id=f"bf16-source:{rp}",
                row_index=0,
                image_width=width,
                image_height=height,
            )
            spans = _row_token_spans(labels)
            if len(spans) != len(parsed.predictions):
                raise HFNativeOneImageOwnerError(
                    "BF16-native Source parser rows lack exact token spans"
                )
            base = dict(request.source_audits[rp])
            provenance = base.get("provenance")
            if not isinstance(provenance, Mapping):
                raise HFNativeOneImageOwnerError(
                    "fp32 Source audit lacks canonical provenance"
                )
            predictions = tuple(
                {
                    "generated_order": int(prediction["generated_order"]),
                    "description": str(prediction["description"]),
                    "bbox": list(prediction["bbox"]),
                    "token_start": spans[index][0],
                    "token_end": spans[index][1],
                }
                for index, prediction in enumerate(parsed.predictions)
            )
            result[rp] = {
                **base,
                "repetition_penalty": rp,
                "generated_token_ids": list(generated),
                "predictions": list(predictions),
                "parser": parsed.parser_policy,
                "parser_status": parsed.parse_status,
                "stop_reason": "im_end",
                "terminal_token_index": len(generated) - 1,
                "malformed_row_count": len(parsed.dropped_predictions),
                "surface": "bf16/flash_attention_2",
                "image_width": width,
                "image_height": height,
                "provenance": dict(provenance),
            }
        return result

    for rp, output in request.source_audits.items():
        prompt = output.get("prompt_token_ids")
        generated = output.get("generated_token_ids")
        predictions = output.get("predictions")
        provenance = output.get("provenance")
        if not all(
            isinstance(value, list) for value in (prompt, generated, predictions)
        ) or not isinstance(provenance, Mapping):
            raise HFNativeOneImageOwnerError(
                "Source audit lacks prompt/token/span evidence"
            )
        prompt_values = cast(list[Any], prompt)
        generated_values = cast(list[Any], generated)
        prediction_values = cast(list[Mapping[str, Any]], predictions)
        width_value = output.get("image_width", provenance.get("image_width"))
        height_value = output.get("image_height", provenance.get("image_height"))
        if width_value is not None or height_value is not None:
            if (
                isinstance(width_value, bool)
                or not isinstance(width_value, int)
                or width_value <= 0
                or isinstance(height_value, bool)
                or not isinstance(height_value, int)
                or height_value <= 0
            ):
                raise HFNativeOneImageOwnerError(
                    "Source audit image dimensions are malformed"
                )
            candidate_dimensions = (width_value, height_value)
            if (
                source_image_dimensions is not None
                and source_image_dimensions != candidate_dimensions
            ):
                raise HFNativeOneImageOwnerError(
                    "Source audits disagree on image dimensions"
                )
            source_image_dimensions = candidate_dimensions
        checkpoint_sha = str(provenance.get("checkpoint_payload_sha256"))
        checkpoint_hashes.add(checkpoint_sha)
        runtime_identity = output.get("hf_runtime_identity")
        source_runtime_identities.append(
            cast(Mapping[str, object], runtime_identity)
            if isinstance(runtime_identity, Mapping)
            else {}
        )
        source_checkpoint_payload_sha256s.append(checkpoint_sha)
        source_checkpoint_paths.append(str(provenance.get("checkpoint_path")))
        source_tokenizer_sha256s.append(str(provenance.get("tokenizer_sha256")))
        source_prompt_sha256s.append(json_sha256(prompt_values))
        source_image_sha256s.append(str(provenance.get("image_sha256")))
        source_manifest_sha256s.append(str(provenance.get("manifest_sha256")))
        source_identity = provenance.get("source_checkpoint_identity")
        source_identity_mapping = (
            cast(Mapping[str, object], source_identity)
            if isinstance(source_identity, Mapping)
            else {}
        )
        source_adapter_sha256s.append(
            str(source_identity_mapping.get("adapter_sha256"))
        )
        source_embedding_delta_sha256s.append(
            str(source_identity_mapping.get("special_embedding_sha256"))
        )
        source_base_model_paths.append(
            str(source_identity_mapping.get("base_model_path"))
        )
        matched = _match_prefix(
            manifest_image,
            prediction_values,
            duplicate_iou_threshold=matcher.duplicate_iou_threshold,
            owner_iou_threshold=matcher.owner_iou_threshold,
        )
        canonical_baselines.append(
            CanonicalSourceBaselineReceipt(
                surface=SOURCE_SURFACE,
                repetition_penalty=float(rp),
                canonical_payload=cast(Mapping[str, object], output),
                owner_map=cast(Mapping[str, object], matched.get("owner_matches", {})),
            )
        )
        source_surface_snapshots.append(
            (
                float(rp),
                tuple(int(value) for value in generated_values),
                tuple(prediction_values),
                cast(Mapping[str, Any], matched),
            )
        )
        audit_hashes.append((rp, json_sha256(output)))
    training_source_audits = _build_bf16_source_audits()
    for rp, output in training_source_audits.items():
        prompt = output.get("prompt_token_ids")
        generated = output.get("generated_token_ids")
        predictions = output.get("predictions")
        provenance = output.get("provenance")
        if not all(
            isinstance(value, list) for value in (prompt, generated, predictions)
        ) or not isinstance(provenance, Mapping):
            raise HFNativeOneImageOwnerError(
                "BF16-native Source audit lacks prompt/token/span evidence"
            )
        prompt_values = cast(list[Any], prompt)
        generated_values = cast(list[Any], generated)
        prediction_values = cast(list[Mapping[str, Any]], predictions)
        current_predictions = tuple(
            CurrentPrediction(
                generated_order=int(row["generated_order"]),
                category=str(row["description"]),
                bbox=cast(
                    tuple[float, float, float, float],
                    tuple(float(value) for value in row["bbox"]),
                ),
                token_start=int(row["token_start"]),
                token_end=int(row["token_end"]),
            )
            for row in prediction_values
        )
        checkpoint_sha = str(provenance.get("checkpoint_payload_sha256"))
        decode = CurrentDecode(
            image_id=1584,
            trajectory_id=f"bf16-source:{rp}",
            generated_token_ids=tuple(int(value) for value in generated_values),
            predictions=current_predictions,
            parser=str(output.get("parser")),
            parser_status=str(output.get("parser_status")),
            stop_reason=str(output.get("stop_reason")),
            checkpoint=CheckpointIdentity(
                str(provenance.get("checkpoint_path")), checkpoint_sha
            ),
            terminal_token_index=output.get("terminal_token_index"),
            malformed_row_count=int(output.get("malformed_row_count", 0)),
        )
        frontier = _project_image(manifest_image, decode, protected=set())
        matched = _match_prefix(
            manifest_image,
            prediction_values,
            duplicate_iou_threshold=matcher.duplicate_iou_threshold,
            owner_iou_threshold=matcher.owner_iou_threshold,
        )
        canonical_baselines.append(
            CanonicalSourceBaselineReceipt(
                surface=TRAINING_SURFACE,
                repetition_penalty=float(rp),
                canonical_payload=cast(Mapping[str, object], output),
                owner_map=cast(Mapping[str, object], matched.get("owner_matches", {})),
            )
        )
        training_surface_snapshots.append(
            (
                float(rp),
                tuple(int(value) for value in generated_values),
                tuple(prediction_values),
                cast(Mapping[str, Any], matched),
            )
        )
        rows = {row.generated_order: row for row in frontier.rows}
        owner_rows = tuple(
            sorted(
                (
                    SealedOwnerRow(
                        owner_id=owner_id,
                        owner_class=(
                            LEGACY_M_OWNER_CLASS
                            if owners[owner_id].stratum == "M"
                            else TRUSTED_OWNER_CLASS
                        ),
                        token_start=rows[int(receipt["generated_order"])].token_start,
                        token_end=rows[int(receipt["generated_order"])].token_end,
                    )
                    for owner_id, receipt in matched["owner_matches"].items()
                ),
                key=lambda row: (row.token_start, row.token_end, row.owner_id),
            )
        )
        sealed.append(
            SealedSourceDecode(
                1584,
                rp,
                tuple(prompt_values),
                tuple(generated_values),
                owner_rows,
            )
        )
        if rp == 1.0:
            boundaries.append(
                SourceBoundaryInput(
                    frontier,
                    tuple(prompt_values),
                    rp,
                    manifest_image.image_sha256,
                )
            )
        training_audit_hashes.append((rp, json_sha256(output)))
    training_rp1 = next(
        (snapshot for snapshot in training_surface_snapshots if snapshot[0] == 1.0),
        None,
    )
    if training_rp1 is None:
        raise HFNativeOneImageOwnerError("BF16-native Source lacks RP1.0 baseline")
    training_g = {
        str(owner_id)
        for owner_id in training_rp1[3].get("owner_matches", {})
        if str(owner_id) in owners and owners[str(owner_id)].stratum == "G"
    }
    missing_protected_g = set(manifest_image.g_owner_ids) - training_g
    if missing_protected_g:
        raise HFNativeOneImageOwnerError(
            "BF16-native Source is missing protected G owners: "
            + ",".join(sorted(missing_protected_g))
        )
    if len(checkpoint_hashes) != 1:
        raise HFNativeOneImageOwnerError("Source audits differ in checkpoint identity")
    if len(boundaries) != 1:
        raise HFNativeOneImageOwnerError(
            "HF-native compiler lacks one pre-acquisition Source boundary"
        )
    aliases, alias_bank_sha256 = _frozen_alias_bank(cast(Any, request.manifest))
    if not aliases:
        raise HFNativeOneImageOwnerError("HF-native compiler alias bank is empty")
    compiler_image = _build_image_ledger(
        cast(Any, request.manifest_image),
        boundaries[0],
        alias_bank_sha256=alias_bank_sha256,
    )
    if compiler_image.site is None:
        raise HFNativeOneImageOwnerError(
            "HF-native compiler required site is unavailable"
        )
    compiler_decode = next(
        (item for item in sealed if item.repetition_penalty == 1.0),
        None,
    )
    if compiler_decode is None:
        raise HFNativeOneImageOwnerError(
            "HF-native compiler lacks the sealed RP1.0 Source decode"
        )
    compiler_raw_logits = surface.raw_logit_rows(
        compiler_decode,
        (compiler_image.site.generated_token_index,),
    )
    measurement = WitnessMeasurement(decodes=tuple(sealed), surface=surface)
    witness_binding = WitnessBinding(
        unit_id=UNIT_ID,
        source_checkpoint_sha256=checkpoint_hashes.pop(),
        manifest_sha256=getattr(request.config, "manifest_sha256"),
        frozen_before_acquisition=True,
    )
    frozen_bank: dict[str, Any] = {}
    alias_cache: dict[str, CoordinateAliasReconciliation] = {}
    canonical_gt_boxes = {
        str(owner.owner_id): tuple(float(value) for value in owner.bbox)
        for owner in manifest_image.owners
    }

    def _coordinate_bin(label: str) -> int | None:
        match = COORD_TOKEN_PATTERN.fullmatch(label)
        return (
            parse_coord_token(label, field="source_surface.coordinate")
            if match is not None
            else None
        )

    def _diagnostic_failure(
        reason: str,
        *,
        repetition_penalty: float | None = None,
        source_tokens: Sequence[object] = (),
        training_tokens: Sequence[object] = (),
        source_token_ids: Sequence[object] = (),
        training_token_ids: Sequence[object] = (),
        coordinate_roles: Mapping[int, tuple[str, str]] | None = None,
        source_boxes: Mapping[str, Sequence[float]] | None = None,
        training_boxes: Mapping[str, Sequence[float]] | None = None,
        source_owner_rows: Mapping[str, int] | None = None,
        training_owner_rows: Mapping[str, int] | None = None,
        source_membership: Mapping[str, str] | None = None,
        training_membership: Mapping[str, str] | None = None,
        source_protected_g: Collection[str] = (),
        training_protected_g: Collection[str] = (),
    ) -> CoordinateAliasReconciliation:
        source_box_values = source_boxes or {}
        training_box_values = training_boxes or {}
        source_row_values = source_owner_rows or {}
        training_row_values = training_owner_rows or {}
        source_membership_values = source_membership or {}
        training_membership_values = training_membership or {}
        return reconcile_coordinate_alias(
            source_tokens=source_tokens,
            training_tokens=training_tokens,
            source_token_ids=source_token_ids,
            training_token_ids=training_token_ids,
            repetition_penalty=repetition_penalty,
            forced_failure_reason=reason,
            coordinate_roles=coordinate_roles or {},
            source_boxes=source_box_values,
            training_boxes=training_box_values,
            gt_boxes=canonical_gt_boxes,
            owner_match={owner_id: owner_id for owner_id in source_box_values},
            source_owner_rows=source_row_values,
            training_owner_rows=training_row_values,
            source_membership=source_membership_values,
            training_membership=training_membership_values,
            source_protected_g=source_protected_g,
            training_protected_g=training_protected_g,
        )

    def _diagnostic_failure_from_context(
        reason: str, context: _CoordinateAliasDiagnosticContext
    ) -> CoordinateAliasReconciliation:
        values = context.to_kwargs()
        return _diagnostic_failure(
            reason,
            repetition_penalty=cast(float | None, values["repetition_penalty"]),
            source_tokens=cast(Sequence[object], values["source_tokens"]),
            training_tokens=cast(Sequence[object], values["training_tokens"]),
            source_token_ids=cast(Sequence[object], values["source_token_ids"]),
            training_token_ids=cast(Sequence[object], values["training_token_ids"]),
            coordinate_roles=cast(
                Mapping[int, tuple[str, str]], values["coordinate_roles"]
            ),
            source_boxes=cast(Mapping[str, Sequence[float]], values["source_boxes"]),
            training_boxes=cast(
                Mapping[str, Sequence[float]], values["training_boxes"]
            ),
            source_owner_rows=cast(Mapping[str, int], values["source_owner_rows"]),
            training_owner_rows=cast(Mapping[str, int], values["training_owner_rows"]),
            source_membership=cast(Mapping[str, str], values["source_membership"]),
            training_membership=cast(Mapping[str, str], values["training_membership"]),
            source_protected_g=cast(Collection[str], values["source_protected_g"]),
            training_protected_g=cast(Collection[str], values["training_protected_g"]),
        )

    def coordinate_alias_check() -> CoordinateAliasReconciliation:
        cached = alias_cache.get("result")
        if cached is not None:
            return cached
        diagnostic_context = _CoordinateAliasDiagnosticContext()
        try:
            if (
                len(source_surface_snapshots) != 2
                or len(training_surface_snapshots) != 2
            ):
                result = _diagnostic_failure(
                    "independent Source baseline count differs",
                    source_token_ids=(
                        source_surface_snapshots[0][1]
                        if source_surface_snapshots
                        else ()
                    ),
                    training_token_ids=(
                        training_surface_snapshots[0][1]
                        if training_surface_snapshots
                        else ()
                    ),
                )
                alias_cache["result"] = result
                return result
            all_evidence: list[object] = []
            for snapshot_index, source_snapshot in enumerate(source_surface_snapshots):
                source_rp, source_ids_raw, source_predictions, source_match = (
                    source_snapshot
                )
                training_rp, training_ids_raw, _training_predictions, training_match = (
                    training_surface_snapshots[snapshot_index]
                )
                if float(source_rp) != float(training_rp):
                    result = _diagnostic_failure(
                        "independent Source repetition-penalty baselines differ",
                        repetition_penalty=float(source_rp),
                        source_token_ids=source_ids_raw,
                        training_token_ids=training_ids_raw,
                    )
                    alias_cache["result"] = result
                    return result
                source_ids = tuple(source_ids_raw)
                training_ids = tuple(training_ids_raw)
                source_matches = source_match.get("owner_matches", {})
                training_matches = training_match.get("owner_matches", {})
                source_boxes = {
                    str(owner_id): cast(Sequence[float], receipt["bbox"])
                    for owner_id, receipt in source_matches.items()
                }
                training_boxes = {
                    str(owner_id): cast(Sequence[float], receipt["bbox"])
                    for owner_id, receipt in training_matches.items()
                }
                source_owner_rows = {
                    str(owner_id): int(receipt["generated_order"])
                    for owner_id, receipt in source_matches.items()
                }
                training_owner_rows = {
                    str(owner_id): int(receipt["generated_order"])
                    for owner_id, receipt in training_matches.items()
                }
                source_membership = {
                    owner_id: str(owners[owner_id].stratum)
                    for owner_id in source_boxes
                    if owner_id in owners
                }
                training_membership = {
                    owner_id: str(owners[owner_id].stratum)
                    for owner_id in training_boxes
                    if owner_id in owners
                }
                source_protected_g = tuple(
                    owner_id
                    for owner_id, stratum in source_membership.items()
                    if stratum == "G"
                )
                training_protected_g = tuple(
                    owner_id
                    for owner_id, stratum in training_membership.items()
                    if stratum == "G"
                )
                source_labels = _token_labels(source_ids)
                training_labels = _token_labels(training_ids)
                coordinate_roles: dict[int, tuple[str, str]] = {}
                owner_by_order = {
                    int(receipt["generated_order"]): str(owner_id)
                    for owner_id, receipt in source_matches.items()
                }
                for row in source_predictions:
                    row_start = int(row["token_start"])
                    row_end = int(row["token_end"])
                    positions = [
                        position
                        for position in range(row_start, row_end)
                        if 0 <= position < len(source_labels)
                        and _coordinate_bin(source_labels[position]) is not None
                    ]
                    if len(positions) != 4:
                        result = _diagnostic_failure(
                            "Source canonical row lacks four coordinate tokens",
                            repetition_penalty=float(source_rp),
                            source_tokens=source_labels,
                            training_tokens=training_labels,
                            source_token_ids=source_ids,
                            training_token_ids=training_ids,
                            source_boxes=source_boxes,
                            training_boxes=training_boxes,
                            source_owner_rows=source_owner_rows,
                            training_owner_rows=training_owner_rows,
                            source_membership=source_membership,
                            training_membership=training_membership,
                            source_protected_g=source_protected_g,
                            training_protected_g=training_protected_g,
                        )
                        alias_cache["result"] = result
                        return result
                    owner_id = owner_by_order.get(int(row["generated_order"]))
                    if owner_id is not None:
                        coordinate_roles.update(
                            {
                                position: (owner_id, role)
                                for role, position in zip(
                                    ("x1", "y1", "x2", "y2"),
                                    positions,
                                    strict=True,
                                )
                            }
                        )
                diagnostic_context = replace(
                    diagnostic_context,
                    repetition_penalty=float(source_rp),
                    source_tokens=source_labels,
                    training_tokens=training_labels,
                    source_token_ids=source_ids,
                    training_token_ids=training_ids,
                    coordinate_roles=_CoordinateAliasDiagnosticContext._roles(
                        coordinate_roles
                    ),
                    source_boxes=_CoordinateAliasDiagnosticContext._boxes(source_boxes),
                    training_boxes=_CoordinateAliasDiagnosticContext._boxes(
                        training_boxes
                    ),
                    source_owner_rows=_CoordinateAliasDiagnosticContext._rows(
                        source_owner_rows
                    ),
                    training_owner_rows=_CoordinateAliasDiagnosticContext._rows(
                        training_owner_rows
                    ),
                    source_membership=_CoordinateAliasDiagnosticContext._membership(
                        source_membership
                    ),
                    training_membership=_CoordinateAliasDiagnosticContext._membership(
                        training_membership
                    ),
                    source_protected_g=source_protected_g,
                    training_protected_g=training_protected_g,
                )
                result = reconcile_coordinate_alias(
                    source_tokens=source_labels,
                    training_tokens=training_labels,
                    source_token_ids=source_ids,
                    training_token_ids=training_ids,
                    repetition_penalty=float(source_rp),
                    coordinate_roles=coordinate_roles,
                    source_boxes=source_boxes,
                    training_boxes=training_boxes,
                    gt_boxes=canonical_gt_boxes,
                    owner_match={owner_id: owner_id for owner_id in source_boxes},
                    source_owner_rows=source_owner_rows,
                    training_owner_rows=training_owner_rows,
                    source_membership=source_membership,
                    training_membership=training_membership,
                    source_protected_g=source_protected_g,
                    training_protected_g=training_protected_g,
                )
                if not result.admitted:
                    alias_cache["result"] = result
                    return result
                all_evidence.extend(result.evidence)
            result = CoordinateAliasReconciliation(
                admitted=True,
                mismatch_count=0,
                failure_reason=None,
                evidence=tuple(cast(Any, all_evidence)),
            )
        except BaseException as error:
            result = _diagnostic_failure_from_context(
                f"coordinate_alias_checker_error:{type(error).__name__}: {error}",
                diagnostic_context,
            )
        alias_cache["result"] = result
        return result

    def freeze_and_check() -> int:
        frozen_bank["bank"] = measurement.freeze_witness_bank(binding=witness_binding)
        # Cross-surface token/row/owner divergence is diagnostic-only.  The
        # checker is still executed and persisted, but it cannot block the
        # independent BF16 policy baseline after the witness has frozen.
        coordinate_alias_check()
        return 0

    reconciliation = reconcile_source_surface(
        SourceSurfaceReconciliationRequest(
            training_identity=identity,
            source_runtime_identities=tuple(source_runtime_identities),
            source_checkpoint_payload_sha256s=tuple(source_checkpoint_payload_sha256s),
            source_checkpoint_paths=tuple(source_checkpoint_paths),
            training_checkpoint_path=str(
                getattr(
                    getattr(getattr(request.assembly, "plan", None), "source", None),
                    "checkpoint_path",
                    getattr(request.config, "source_checkpoint_path", None),
                )
            ),
            source_adapter_sha256s=tuple(source_adapter_sha256s),
            source_embedding_delta_sha256s=tuple(source_embedding_delta_sha256s),
            source_base_model_paths=tuple(source_base_model_paths),
            training_base_model_path=str(
                getattr(
                    request.config,
                    "base_model_path",
                    getattr(
                        getattr(
                            getattr(request.assembly, "plan", None), "source", None
                        ),
                        "base_model_path",
                        None,
                    ),
                )
            ),
            source_manifest_sha256s=tuple(source_manifest_sha256s),
            manifest_image_sha256=str(
                getattr(request.manifest_image, "image_sha256", None)
            ),
            source_tokenizer_sha256s=tuple(source_tokenizer_sha256s),
            source_prompt_sha256s=tuple(source_prompt_sha256s),
            source_image_sha256s=tuple(source_image_sha256s),
            manifest_sha256=getattr(request.config, "manifest_sha256"),
            image_id=1584,
            source_audit_sha256s=tuple(audit_hashes),
            decodes=tuple(sealed),
            check=freeze_and_check,
            coordinate_alias_check=coordinate_alias_check,
            canonical_baselines=tuple(canonical_baselines),
        )
    )
    if not reconciliation.admitted:
        raise HFNativeOneImageOwnerError(
            reconciliation.failure_reason or "Source surfaces failed reconciliation",
            disposition="source_surface_reconciliation_failure",
            reconciliation_receipt=reconciliation,
        )
    lazy_bank = frozen_bank["bank"]
    bank = freeze_witness_bank_for_post_acquisition(lazy_bank)
    realized_margin_probe = AdmittedPostApplyMarginProbe(
        session=request.session,
        source_parameter_state_sha256=identity.parameter_state_sha256,
        source_decodes=tuple(sealed),
        witness_sites=cast(Any, bank).constraints,
    )
    return PreAcquisitionSourceOwners(
        session_object_id=id(request.session),
        model_object_id=id(getattr(request.assembly, "model")),
        parameter_state_sha256=identity.parameter_state_sha256,
        manifest_sha256=getattr(request.config, "manifest_sha256"),
        image_sha256=identity.image_sha256,
        source_checkpoint_sha256=witness_binding.source_checkpoint_sha256,
        source_audit_sha256s=tuple(training_audit_hashes),
        compiler_source_context=tuple(boundaries),
        witness_bank=bank,
        realized_margin_probe=realized_margin_probe,
        frozen_before_acquisition=True,
        sample_group_count_at_freeze=0,
        replay_group_count_at_freeze=0,
        source_decodes=tuple(sealed),
        compiler_raw_logits=compiler_raw_logits,
        surface_reconciliation=reconciliation,
    )


def admit_repository_hf_native_trajectory(
    request: HFNativeAdmissionRequest,
    source: PreAcquisitionSourceOwners,
    *,
    canonical_projections: tuple[object, ...],
) -> HFNativeTrajectoryAdmission:
    replayed = cast(tuple[GradientReplayGroup, ...], request.replay_groups)
    sampled = cast(tuple[SampledHFGroup, ...], request.sampled_groups)
    ledger = construct_hf_native_one_image_trajectory_ledger(
        manifest=request.manifest,
        manifest_image=request.manifest_image,
        replay_groups=replayed,
        canonical_projections=canonical_projections,
        source_checkpoint_sha256=source.source_checkpoint_sha256,
    )
    return HFNativeTrajectoryAdmission(
        source_owner_sha256=source.content_sha256,
        session_object_id=id(request.session),
        model_object_id=id(getattr(request.assembly, "model")),
        manifest_object_id=id(request.manifest),
        manifest_sha256=source.manifest_sha256,
        image_id=1584,
        image_sha256=source.image_sha256,
        source_checkpoint_sha256=source.source_checkpoint_sha256,
        parameter_state_sha256=sampled[0].identity.parameter_state_sha256,
        sampled_group_sha256s=tuple(group.content_sha256 for group in sampled),
        replay_group_sha256s=tuple(group.content_sha256 for group in replayed),
        parser_projection_sha256=cast(Any, ledger).images[0].parser_projection_sha256,
        trajectory_ledger=ledger,
    )


def admit_repository_hf_native_compiler(
    request: HFNativeAdmissionRequest,
    source: PreAcquisitionSourceOwners,
    trajectory: HFNativeTrajectoryAdmission,
) -> HFNativeCompilerAdmission:
    from scripts.research.human13_greedy_compiler import (
        admit_hf_native_compiler_compact_logits,
        construct_hf_native_one_image_compiler_ledger,
    )

    boundaries = tuple(cast(Any, source.compiler_source_context))
    if len(boundaries) != 1:
        raise HFNativeOneImageOwnerError(
            "HF-native compiler lacks one pre-acquisition Source boundary"
        )
    boundary = boundaries[0]
    ledger = construct_hf_native_one_image_compiler_ledger(
        cast(Any, request.manifest),
        cast(Any, request.manifest_image),
        boundary,
        cast(Any, trajectory.trajectory_ledger),
    )
    raw_logits = source.compiler_raw_logits
    if raw_logits is None:
        raise HFNativeOneImageOwnerError(
            "HF-native compiler graph row was not frozen before acquisition"
        )
    compact = admit_hf_native_compiler_compact_logits(
        ledger,
        boundary,
        raw_logits=cast(Any, raw_logits),
    )
    return HFNativeCompilerAdmission(
        source_owner_sha256=source.content_sha256,
        trajectory_admission_sha256=trajectory.content_sha256,
        session_object_id=id(request.session),
        model_object_id=id(getattr(request.assembly, "model")),
        manifest_object_id=id(request.manifest),
        manifest_sha256=source.manifest_sha256,
        image_id=1584,
        image_sha256=source.image_sha256,
        source_checkpoint_sha256=source.source_checkpoint_sha256,
        source_boundary_sha256s=(boundary.source_decode_sha256,),
        compiler_ledger=ledger,
        compiler_compact_logits=compact,
    )


def build_repository_hf_native_owner(
    *,
    prepare_source_owner: Callable[
        [SourceOwnerRequest], PreAcquisitionSourceOwners
    ] = prepare_repository_source_owners,
    admit_trajectory: Callable[
        [HFNativeAdmissionRequest, PreAcquisitionSourceOwners],
        HFNativeTrajectoryAdmission,
    ]
    | None = None,
    admit_compiler: Callable[
        [
            HFNativeAdmissionRequest,
            PreAcquisitionSourceOwners,
            HFNativeTrajectoryAdmission,
        ],
        HFNativeCompilerAdmission,
    ]
    | None = None,
    canonical_projection_provider: Callable[
        [HFNativeAdmissionRequest, PreAcquisitionSourceOwners], tuple[object, ...]
    ]
    | None = None,
) -> RepositoryHFNativeOneImageOwner | None:
    """Compose the fail-closed repository bridge from same-session producers."""

    if admit_trajectory is None and canonical_projection_provider is None:

        def project_canonical_replays(
            request: HFNativeAdmissionRequest,
            _source: PreAcquisitionSourceOwners,
        ) -> tuple[object, ...]:
            projector = getattr(request.session, "canonical_replay_projections", None)
            if not callable(projector):
                raise HFNativeOneImageOwnerError(
                    "shared session lacks the HF-native canonical projector"
                )
            try:
                projections = projector(
                    request.replay_groups,
                    request.manifest,
                    request.manifest_image,
                )
            except HFNativeOneImageOwnerError:
                raise
            except Exception as error:
                raise HFNativeOneImageOwnerError(
                    "canonical replay projection failed: "
                    f"{type(error).__name__}: {error}",
                    disposition="canonical_projection_failure",
                ) from error
            if not isinstance(projections, tuple):
                raise HFNativeOneImageOwnerError(
                    "shared session returned an untyped canonical projection batch"
                )
            return projections

        canonical_projection_provider = project_canonical_replays

    if admit_trajectory is None and canonical_projection_provider is not None:

        def admit_native_trajectory(
            request: HFNativeAdmissionRequest,
            source: PreAcquisitionSourceOwners,
        ) -> HFNativeTrajectoryAdmission:
            return admit_repository_hf_native_trajectory(
                request,
                source,
                canonical_projections=canonical_projection_provider(request, source),
            )

        admit_trajectory = admit_native_trajectory
    if admit_trajectory is not None and admit_compiler is None:
        admit_compiler = admit_repository_hf_native_compiler
    if admit_trajectory is None or admit_compiler is None:
        return None

    def admit_hf_owners(
        request: HFNativeAdmissionRequest,
        source: PreAcquisitionSourceOwners,
    ) -> HFNativeOneImageAdmission:
        sampled = cast(tuple[SampledHFGroup, ...], request.sampled_groups)
        replayed = cast(tuple[GradientReplayGroup, ...], request.replay_groups)
        identities = tuple(group.identity for group in sampled)
        if len(set(identities)) != 1:
            raise HFNativeOneImageOwnerError(
                "HF-native groups differ in shared-surface identity"
            )
        identity = identities[0]
        model = getattr(request.assembly, "model", None)
        manifest_sha256 = getattr(request.config, "manifest_sha256", None)
        image_sha256 = getattr(request.manifest_image, "image_sha256", None)
        if (
            model is None
            or identity.model_object_id != id(model)
            or source.session_object_id != id(request.session)
            or source.model_object_id != id(model)
            or source.parameter_state_sha256 != identity.parameter_state_sha256
            or source.manifest_sha256 != manifest_sha256
            or source.image_sha256 != image_sha256
        ):
            raise HFNativeOneImageOwnerError(
                "HF-native request differs from frozen Source/model/session lineage"
            )
        sampled_hashes = tuple(group.content_sha256 for group in sampled)
        replay_hashes = tuple(group.content_sha256 for group in replayed)
        if tuple(group.group_index for group in sampled) != (0, 1, 2, 3):
            raise HFNativeOneImageOwnerError(
                "HF-native sampled groups differ from the four replay groups"
            )
        if (
            tuple(group.sampled_group.content_sha256 for group in replayed)
            != sampled_hashes
        ):
            raise HFNativeOneImageOwnerError(
                "HF-native replay groups differ from sampled group lineage"
            )
        if set(request.replay_logprob_tensors) != set(replay_hashes):
            raise HFNativeOneImageOwnerError(
                "HF-native graph tensors differ from the four replay groups"
            )

        trajectory = admit_trajectory(request, source)
        if type(trajectory) is not HFNativeTrajectoryAdmission:
            raise HFNativeOneImageOwnerError(
                "HF-native trajectory owner returned an untyped admission"
            )
        expected_trajectory = (
            source.content_sha256,
            id(request.session),
            id(model),
            id(request.manifest),
            manifest_sha256,
            1584,
            image_sha256,
            source.source_checkpoint_sha256,
            identity.parameter_state_sha256,
            sampled_hashes,
            replay_hashes,
        )
        observed_trajectory = (
            trajectory.source_owner_sha256,
            trajectory.session_object_id,
            trajectory.model_object_id,
            trajectory.manifest_object_id,
            trajectory.manifest_sha256,
            trajectory.image_id,
            trajectory.image_sha256,
            trajectory.source_checkpoint_sha256,
            trajectory.parameter_state_sha256,
            trajectory.sampled_group_sha256s,
            trajectory.replay_group_sha256s,
        )
        if observed_trajectory != expected_trajectory:
            raise HFNativeOneImageOwnerError(
                "HF-native trajectory admission changed live lineage"
            )
        ledger = cast(Any, trajectory.trajectory_ledger)
        request_ids = tuple(
            item.request_id for group in sampled for item in group.requests
        )
        ledger_request_ids = tuple(
            item.request_id for image in ledger.images for item in image.trajectories
        )
        if ledger_request_ids != request_ids:
            raise HFNativeOneImageOwnerError(
                "HF-native canonical parser spans differ from replay requests"
            )

        compiler = admit_compiler(request, source, trajectory)
        if type(compiler) is not HFNativeCompilerAdmission:
            raise HFNativeOneImageOwnerError(
                "HF-native compiler owner returned an untyped admission"
            )
        boundaries = tuple(cast(Any, source.compiler_source_context))
        boundary_hashes = tuple(
            getattr(boundary, "source_decode_sha256", None) for boundary in boundaries
        )
        expected_compiler = (
            source.content_sha256,
            trajectory.content_sha256,
            id(request.session),
            id(model),
            id(request.manifest),
            manifest_sha256,
            1584,
            image_sha256,
            source.source_checkpoint_sha256,
            boundary_hashes,
        )
        observed_compiler = (
            compiler.source_owner_sha256,
            compiler.trajectory_admission_sha256,
            compiler.session_object_id,
            compiler.model_object_id,
            compiler.manifest_object_id,
            compiler.manifest_sha256,
            compiler.image_id,
            compiler.image_sha256,
            compiler.source_checkpoint_sha256,
            compiler.source_boundary_sha256s,
        )
        compiler_ledger = cast(Any, compiler.compiler_ledger)
        if (
            observed_compiler != expected_compiler
            or compiler_ledger.acquisition_sha256 != ledger.acquisition_sha256
            or compiler_ledger.trajectory_credit_sha256 != ledger.content_sha256
        ):
            raise HFNativeOneImageOwnerError(
                "HF-native compiler admission changed Source/trajectory lineage"
            )

        from scripts.research.human13_one_image_services import (
            AdmittedTask5RuntimeEvidence,
        )

        evidence = AdmittedTask5RuntimeEvidence(
            trajectory_ledger=trajectory.trajectory_ledger,
            compiler_ledger=compiler.compiler_ledger,
            compiler_compact_logits=compiler.compiler_compact_logits,
            witness_bank=source.witness_bank,
            realized_margin_probe=source.realized_margin_probe,
        )
        return HFNativeOneImageAdmission(
            source_owner_sha256=source.content_sha256,
            session_object_id=id(request.session),
            model_object_id=id(model),
            sampled_group_sha256s=sampled_hashes,
            replay_group_sha256s=replay_hashes,
            runtime_evidence=evidence,
        )

    return RepositoryHFNativeOneImageOwner(
        prepare_source_owner=prepare_source_owner,
        admit_hf_owners=admit_hf_owners,
    )


__all__ = [
    "AdmittedPostApplyMarginProbe",
    "HFNativeAdmissionRequest",
    "HFNativeCompilerAdmission",
    "HFNativeOneImageAdmission",
    "HFNativeOneImageOwner",
    "HFNativeOneImageOwnerError",
    "HFNativeTrajectoryAdmission",
    "PreAcquisitionSourceOwners",
    "RepositoryHFNativeOneImageOwner",
    "SourceOwnerRequest",
    "admit_repository_hf_native_compiler",
    "admit_repository_hf_native_trajectory",
    "build_repository_hf_native_owner",
    "construct_hf_native_one_image_trajectory_ledger",
    "freeze_witness_bank_for_post_acquisition",
    "hf_native_request_evidence_sha256",
    "prepare_repository_source_owners",
]
