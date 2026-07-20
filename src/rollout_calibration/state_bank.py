"""Immutable reviewed rollout state-bank contracts for calibration training.

This module deliberately validates declarations instead of deriving scientific
labels.  The offline assembler joins exact rollout token evidence with an
explicit review decision; training consumes only the resulting frozen records.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from PIL import Image

from src.common.errors import ArtifactContractError
from src.config.fingerprint import sha256_file, sha256_json
from src.data import ImageRef, RawExample, RawObject, SourceProvenance
from src.data.examples import freeze_json
from src.data.geometry import validate_bbox_bins
from src.data.images import image_stat_fingerprint
from src.inference.backend import token_ids_sha256


STATE_BANK_SCHEMA_VERSION = "coordexp.rollout_calibration.state_bank.v1"
STATE_BANK_RECORDS_NAME = "records.jsonl"
STATE_BANK_MANIFEST_NAME = "manifest.json"

BLIND_IMAGE_IDS = frozenset(
    {1584, 2685, 4134, 5001, 6040, 7511, 10707, 13348, 13923, 14038, 14439, 16228}
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = frozenset({"train", "eval"})
_REVIEW_STATUSES = frozenset({"trusted", "unknown", "ambiguous"})
_ROLES = frozenset({"positive", "harmful", "diagnostic"})
_HARMFUL_KINDS = frozenset({"duplicate", "premature_terminal"})
_COVERAGE_STATUSES = frozenset({"uncovered", "covered", "unknown"})
_PREFIX_COVERAGE_STATUSES = frozenset({"empty", "resolved", "unresolved"})
_TOKEN_TYPES = frozenset({"desc_text", "schema", "coordinate", "eos"})
_COORDINATE_AXES = {
    "x1": "horizontal",
    "y1": "vertical",
    "x2": "horizontal",
    "y2": "vertical",
}
_COORDINATE_ORDER = ("x1", "y1", "x2", "y2")
_GENERATION_MODES = frozenset({"greedy", "sampled"})


@dataclass(frozen=True)
class ReviewProvenance:
    source: str
    reviewer: str
    confidence: str
    comment: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, field: str
    ) -> "ReviewProvenance":
        checked = _mapping(value, field=field)
        _require_exact_keys(
            checked,
            {"source", "reviewer", "confidence", "comment"},
            field=field,
        )
        return cls(
            source=_string(checked["source"], field=f"{field}.source"),
            reviewer=_string(checked["reviewer"], field=f"{field}.reviewer"),
            confidence=_string(checked["confidence"], field=f"{field}.confidence"),
            comment=_string_allow_empty(checked["comment"], field=f"{field}.comment"),
        )

    def to_artifact_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "reviewer": self.reviewer,
            "confidence": self.confidence,
            "comment": self.comment,
        }


@dataclass(frozen=True)
class GenerationProvenance:
    mode: str
    seed: int
    temperature: float
    top_p: float
    repetition_penalty: float
    checkpoint_id: str
    prompt_token_ids_sha256: str
    prefix_token_ids_sha256: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, field: str
    ) -> "GenerationProvenance":
        checked = _mapping(value, field=field)
        _require_exact_keys(
            checked,
            {
                "mode",
                "seed",
                "temperature",
                "top_p",
                "repetition_penalty",
                "checkpoint_id",
                "prompt_token_ids_sha256",
                "prefix_token_ids_sha256",
            },
            field=field,
        )
        mode = _choice(checked["mode"], _GENERATION_MODES, field=f"{field}.mode")
        temperature = _finite_float(
            checked["temperature"], field=f"{field}.temperature", minimum=0.0
        )
        top_p = _finite_float(
            checked["top_p"], field=f"{field}.top_p", minimum=0.0, maximum=1.0
        )
        repetition_penalty = _finite_float(
            checked["repetition_penalty"],
            field=f"{field}.repetition_penalty",
            minimum=0.0,
        )
        if mode == "greedy" and temperature != 0.0:
            _fail(
                "state_bank.greedy_temperature",
                "producer-declared greedy generation requires temperature zero",
                field=field,
                temperature=temperature,
            )
        if mode == "sampled" and temperature <= 0.0:
            _fail(
                "state_bank.sampled_temperature",
                "producer-declared sampled generation requires positive temperature",
                field=field,
                temperature=temperature,
            )
        if top_p <= 0.0 or repetition_penalty <= 0.0:
            _fail(
                "state_bank.generation_policy_range",
                "top-p and repetition penalty must be positive",
                field=field,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        checkpoint_id = _string(
            checked["checkpoint_id"], field=f"{field}.checkpoint_id"
        )
        prompt_hash = _string(
            checked["prompt_token_ids_sha256"],
            field=f"{field}.prompt_token_ids_sha256",
        )
        prefix_hash = _string(
            checked["prefix_token_ids_sha256"],
            field=f"{field}.prefix_token_ids_sha256",
        )
        _require_sha256(checkpoint_id, field=f"{field}.checkpoint_id")
        _require_sha256(prompt_hash, field=f"{field}.prompt_token_ids_sha256")
        _require_sha256(prefix_hash, field=f"{field}.prefix_token_ids_sha256")
        return cls(
            mode=mode,
            seed=_require_nonnegative_int(checked["seed"], field=f"{field}.seed"),
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            checkpoint_id=checkpoint_id,
            prompt_token_ids_sha256=prompt_hash,
            prefix_token_ids_sha256=prefix_hash,
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "checkpoint_id": self.checkpoint_id,
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "prefix_token_ids_sha256": self.prefix_token_ids_sha256,
        }


@dataclass(frozen=True)
class PrefixCoveredOwnerProof:
    prefix_object_row_index: int
    owner_id: str
    review_provenance: ReviewProvenance

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, field: str
    ) -> "PrefixCoveredOwnerProof":
        checked = _mapping(value, field=field)
        _require_exact_keys(
            checked,
            {"prefix_object_row_index", "owner_id", "review_provenance"},
            field=field,
        )
        return cls(
            prefix_object_row_index=_require_nonnegative_int(
                checked["prefix_object_row_index"],
                field=f"{field}.prefix_object_row_index",
            ),
            owner_id=_string(checked["owner_id"], field=f"{field}.owner_id"),
            review_provenance=ReviewProvenance.from_mapping(
                _mapping(
                    checked["review_provenance"], field=f"{field}.review_provenance"
                ),
                field=f"{field}.review_provenance",
            ),
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "prefix_object_row_index": self.prefix_object_row_index,
            "owner_id": self.owner_id,
            "review_provenance": self.review_provenance.to_artifact_dict(),
        }


@dataclass(frozen=True)
class CheckpointIdentity:
    adapter_fingerprint: str
    embedding_delta_fingerprint: str
    base_config_sha256: str
    tokenizer_sha256: str
    token_identity_sha256: str
    special_token_identity_sha256: str
    processor_identity_sha256: str

    def __post_init__(self) -> None:
        for field, value in self.to_artifact_dict().items():
            _require_sha256(value, field=f"source_checkpoint.{field}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CheckpointIdentity":
        checked = _mapping(value, field="source_checkpoint")
        _require_exact_keys(
            checked,
            {
                "adapter_fingerprint",
                "embedding_delta_fingerprint",
                "base_config_sha256",
                "tokenizer_sha256",
                "token_identity_sha256",
                "special_token_identity_sha256",
                "processor_identity_sha256",
            },
            field="source_checkpoint",
        )
        return cls(
            **{
                key: _string(checked[key], field=f"source_checkpoint.{key}")
                for key in checked
            }
        )

    def to_artifact_dict(self) -> dict[str, str]:
        return {
            "adapter_fingerprint": self.adapter_fingerprint,
            "embedding_delta_fingerprint": self.embedding_delta_fingerprint,
            "base_config_sha256": self.base_config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "token_identity_sha256": self.token_identity_sha256,
            "special_token_identity_sha256": self.special_token_identity_sha256,
            "processor_identity_sha256": self.processor_identity_sha256,
        }


@dataclass(frozen=True)
class ImageIdentity:
    image_id: int
    path: Path
    width: int
    height: int
    content_sha256: str

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.image_id, field="image.image_id")
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        _require_positive_int(self.width, field="image.width")
        _require_positive_int(self.height, field="image.height")
        _require_sha256(self.content_sha256, field="image.content_sha256")
        _reject_blind_image_id(self.image_id, field="image.image_id")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ImageIdentity":
        checked = _mapping(value, field="image")
        _require_exact_keys(
            checked,
            {"image_id", "path", "width", "height", "content_sha256"},
            field="image",
        )
        return cls(
            image_id=_image_id(checked["image_id"], field="image.image_id"),
            path=Path(_string(checked["path"], field="image.path")),
            width=_require_positive_int(checked["width"], field="image.width"),
            height=_require_positive_int(checked["height"], field="image.height"),
            content_sha256=_string(
                checked["content_sha256"], field="image.content_sha256"
            ),
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class PhysicalEntity:
    entity_id: str
    category: str
    entity_trusted: bool
    geometry_trusted: bool
    reference_bbox: tuple[int, int, int, int]
    review_source: str
    reviewer: str
    review_confidence: str
    comment: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, field: str) -> "PhysicalEntity":
        checked = _mapping(value, field=field)
        _require_exact_keys(
            checked,
            {
                "entity_id",
                "category",
                "entity_trusted",
                "geometry_trusted",
                "reference_bbox",
                "review_source",
                "reviewer",
                "review_confidence",
                "comment",
            },
            field=field,
        )
        return cls(
            entity_id=_string(checked["entity_id"], field=f"{field}.entity_id"),
            category=_string(checked["category"], field=f"{field}.category"),
            entity_trusted=_bool(
                checked["entity_trusted"], field=f"{field}.entity_trusted"
            ),
            geometry_trusted=_bool(
                checked["geometry_trusted"], field=f"{field}.geometry_trusted"
            ),
            reference_bbox=validate_bbox_bins(
                checked["reference_bbox"], field=f"{field}.reference_bbox"
            ),
            review_source=_string(
                checked["review_source"], field=f"{field}.review_source"
            ),
            reviewer=_string(checked["reviewer"], field=f"{field}.reviewer"),
            review_confidence=_string(
                checked["review_confidence"], field=f"{field}.review_confidence"
            ),
            comment=_string_allow_empty(checked["comment"], field=f"{field}.comment"),
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "category": self.category,
            "entity_trusted": self.entity_trusted,
            "geometry_trusted": self.geometry_trusted,
            "reference_bbox": list(self.reference_bbox),
            "review_source": self.review_source,
            "reviewer": self.reviewer,
            "review_confidence": self.review_confidence,
            "comment": self.comment,
        }


@dataclass(frozen=True)
class SelectedSite:
    candidate_token_offset: int
    intended_token_type: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, field: str) -> "SelectedSite":
        checked = _mapping(value, field=field)
        _require_exact_keys(
            checked,
            {"candidate_token_offset", "intended_token_type"},
            field=field,
        )
        token_type = _choice(
            checked["intended_token_type"],
            _TOKEN_TYPES,
            field=f"{field}.intended_token_type",
        )
        return cls(
            candidate_token_offset=_require_nonnegative_int(
                checked["candidate_token_offset"],
                field=f"{field}.candidate_token_offset",
            ),
            intended_token_type=token_type,
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "candidate_token_offset": self.candidate_token_offset,
            "intended_token_type": self.intended_token_type,
        }


@dataclass(frozen=True)
class CoordinateBoundaryObservation:
    coordinate: str
    tolerance_axis: str
    candidate_token_offset: int
    actual_coordinate_value: int
    acceptable_coordinate_values: tuple[int, ...]
    review_provenance: ReviewProvenance

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, field: str
    ) -> "CoordinateBoundaryObservation":
        checked = _mapping(value, field=field)
        _require_exact_keys(
            checked,
            {
                "coordinate",
                "tolerance_axis",
                "candidate_token_offset",
                "actual_coordinate_value",
                "acceptable_coordinate_values",
                "review_provenance",
            },
            field=field,
        )
        coordinate = _choice(
            checked["coordinate"],
            frozenset(_COORDINATE_AXES),
            field=f"{field}.coordinate",
        )
        tolerance_axis = _choice(
            checked["tolerance_axis"],
            frozenset({"horizontal", "vertical"}),
            field=f"{field}.tolerance_axis",
        )
        if tolerance_axis != _COORDINATE_AXES[coordinate]:
            _fail(
                "state_bank.coordinate_axis",
                "coordinate acceptable set uses the wrong tolerance axis",
                field=field,
                coordinate=coordinate,
                tolerance_axis=tolerance_axis,
                expected_axis=_COORDINATE_AXES[coordinate],
            )
        acceptable = _coordinate_values(
            checked["acceptable_coordinate_values"],
            field=f"{field}.acceptable_coordinate_values",
        )
        actual = _coordinate_value(
            checked["actual_coordinate_value"],
            field=f"{field}.actual_coordinate_value",
        )
        return cls(
            coordinate=coordinate,
            tolerance_axis=tolerance_axis,
            candidate_token_offset=_require_nonnegative_int(
                checked["candidate_token_offset"],
                field=f"{field}.candidate_token_offset",
            ),
            actual_coordinate_value=actual,
            acceptable_coordinate_values=acceptable,
            review_provenance=ReviewProvenance.from_mapping(
                _mapping(
                    checked["review_provenance"], field=f"{field}.review_provenance"
                ),
                field=f"{field}.review_provenance",
            ),
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate,
            "tolerance_axis": self.tolerance_axis,
            "candidate_token_offset": self.candidate_token_offset,
            "actual_coordinate_value": self.actual_coordinate_value,
            "acceptable_coordinate_values": list(self.acceptable_coordinate_values),
            "review_provenance": self.review_provenance.to_artifact_dict(),
        }


@dataclass(frozen=True)
class CoordinateDecision:
    owner_id: str
    observations: tuple[CoordinateBoundaryObservation, ...]

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, field: str
    ) -> "CoordinateDecision":
        checked = _mapping(value, field=field)
        _require_exact_keys(checked, {"owner_id", "observations"}, field=field)
        observations = tuple(
            CoordinateBoundaryObservation.from_mapping(
                item, field=f"{field}.observations[{index}]"
            )
            for index, item in enumerate(
                _sequence(checked["observations"], field=f"{field}.observations")
            )
        )
        if not observations:
            _fail(
                "state_bank.coordinate_observations_empty",
                "coordinate decision requires the ordered prefix through the first wrong boundary",
                field=field,
            )
        expected_coordinates = _COORDINATE_ORDER[: len(observations)]
        actual_coordinates = tuple(item.coordinate for item in observations)
        if actual_coordinates != expected_coordinates:
            _fail(
                "state_bank.coordinate_observation_order",
                "coordinate observations must be the ordered x1,y1,x2,y2 prefix",
                field=field,
                expected=list(expected_coordinates),
                actual=list(actual_coordinates),
            )
        offsets = tuple(item.candidate_token_offset for item in observations)
        if tuple(sorted(offsets)) != offsets or len(set(offsets)) != len(offsets):
            _fail(
                "state_bank.coordinate_observation_offsets",
                "coordinate observation token offsets must be unique and increasing",
                field=field,
                offsets=list(offsets),
            )
        for observation in observations[:-1]:
            if (
                observation.actual_coordinate_value
                not in observation.acceptable_coordinate_values
            ):
                _fail(
                    "state_bank.coordinate_earlier_wrong",
                    "every coordinate before the selected boundary must be accepted",
                    field=field,
                    coordinate=observation.coordinate,
                    actual_coordinate_value=observation.actual_coordinate_value,
                )
        selected = observations[-1]
        if selected.actual_coordinate_value in selected.acceptable_coordinate_values:
            _fail(
                "state_bank.coordinate_final_accepted",
                "the final stored boundary must be the first wrong coordinate",
                field=field,
                coordinate=selected.coordinate,
                actual_coordinate_value=selected.actual_coordinate_value,
            )
        return cls(
            owner_id=_string(checked["owner_id"], field=f"{field}.owner_id"),
            observations=observations,
        )

    @property
    def selected_observation(self) -> CoordinateBoundaryObservation:
        return self.observations[-1]

    @property
    def coordinate(self) -> str:
        return self.selected_observation.coordinate

    @property
    def tolerance_axis(self) -> str:
        return self.selected_observation.tolerance_axis

    @property
    def candidate_token_offset(self) -> int:
        return self.selected_observation.candidate_token_offset

    @property
    def actual_wrong_coordinate_value(self) -> int:
        return self.selected_observation.actual_coordinate_value

    @property
    def acceptable_coordinate_values(self) -> tuple[int, ...]:
        return self.selected_observation.acceptable_coordinate_values

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "observations": [item.to_artifact_dict() for item in self.observations],
        }


@dataclass(frozen=True)
class StateBankCandidate:
    candidate_id: str
    role: str
    harmful_kind: str | None
    token_ids: tuple[int, ...]
    token_ids_sha256: str
    physical_owner_id: str | None
    coverage_status: str
    entity_review_status: str
    geometry_review_status: str
    entity_eligible: bool
    geometry_eligible: bool
    owner_resolution_interval: tuple[int, int] | None
    coordinate_decision: CoordinateDecision | None
    selected_sites: tuple[SelectedSite, ...]
    generation_provenance: GenerationProvenance
    evidence_text: str | None

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, field: str
    ) -> "StateBankCandidate":
        checked = _mapping(value, field=field)
        _require_exact_keys(
            checked,
            {
                "candidate_id",
                "role",
                "harmful_kind",
                "token_ids",
                "token_ids_sha256",
                "physical_owner_id",
                "coverage_status",
                "entity_review_status",
                "geometry_review_status",
                "entity_eligible",
                "geometry_eligible",
                "owner_resolution_interval",
                "coordinate_decision",
                "selected_sites",
                "generation_provenance",
                "evidence_text",
            },
            field=field,
        )
        token_ids = _token_ids(checked["token_ids"], field=f"{field}.token_ids")
        declared_hash = _string(
            checked["token_ids_sha256"], field=f"{field}.token_ids_sha256"
        )
        _validate_token_hash(token_ids, declared_hash, field=f"{field}.token_ids")
        physical_owner_id = _optional_string(
            checked["physical_owner_id"], field=f"{field}.physical_owner_id"
        )
        interval = _optional_interval(
            checked["owner_resolution_interval"],
            field=f"{field}.owner_resolution_interval",
            upper_bound=len(token_ids),
        )
        sites = _selected_sites(
            checked["selected_sites"],
            field=f"{field}.selected_sites",
            token_count=len(token_ids),
        )
        coordinate = (
            None
            if checked["coordinate_decision"] is None
            else CoordinateDecision.from_mapping(
                _mapping(
                    checked["coordinate_decision"], field=f"{field}.coordinate_decision"
                ),
                field=f"{field}.coordinate_decision",
            )
        )
        candidate = cls(
            candidate_id=_string(
                checked["candidate_id"], field=f"{field}.candidate_id"
            ),
            role=_choice(checked["role"], _ROLES, field=f"{field}.role"),
            harmful_kind=_optional_choice(
                checked["harmful_kind"], _HARMFUL_KINDS, field=f"{field}.harmful_kind"
            ),
            token_ids=token_ids,
            token_ids_sha256=declared_hash,
            physical_owner_id=physical_owner_id,
            coverage_status=_choice(
                checked["coverage_status"],
                _COVERAGE_STATUSES,
                field=f"{field}.coverage_status",
            ),
            entity_review_status=_choice(
                checked["entity_review_status"],
                _REVIEW_STATUSES,
                field=f"{field}.entity_review_status",
            ),
            geometry_review_status=_choice(
                checked["geometry_review_status"],
                _REVIEW_STATUSES,
                field=f"{field}.geometry_review_status",
            ),
            entity_eligible=_bool(
                checked["entity_eligible"], field=f"{field}.entity_eligible"
            ),
            geometry_eligible=_bool(
                checked["geometry_eligible"], field=f"{field}.geometry_eligible"
            ),
            owner_resolution_interval=interval,
            coordinate_decision=coordinate,
            selected_sites=sites,
            generation_provenance=GenerationProvenance.from_mapping(
                _mapping(
                    checked["generation_provenance"],
                    field=f"{field}.generation_provenance",
                ),
                field=f"{field}.generation_provenance",
            ),
            evidence_text=_optional_string(
                checked["evidence_text"], field=f"{field}.evidence_text"
            ),
        )
        candidate._validate_semantics(field=field)
        return candidate

    def _validate_semantics(self, *, field: str) -> None:
        if self.role == "harmful" and self.harmful_kind is None:
            _fail(
                "state_bank.harmful_kind_missing",
                "harmful candidate requires harmful_kind",
                field=field,
            )
        if self.role != "harmful" and self.harmful_kind is not None:
            _fail(
                "state_bank.harmful_kind_role",
                "only harmful candidates may declare harmful_kind",
                field=field,
            )
        if self.entity_review_status != "trusted" and self.entity_eligible:
            _fail(
                "state_bank.entity_unknown_eligible",
                "unknown or ambiguous entity review must have zero entity eligibility",
                field=field,
            )
        if self.geometry_review_status != "trusted" and self.geometry_eligible:
            _fail(
                "state_bank.geometry_unknown_eligible",
                "unknown or ambiguous geometry review must have zero geometry eligibility",
                field=field,
            )
        site_by_offset = {
            site.candidate_token_offset: site for site in self.selected_sites
        }
        if self.entity_eligible:
            if self.role not in {"positive", "harmful"}:
                _fail(
                    "state_bank.entity_role",
                    "entity-eligible candidate must be positive or harmful",
                    field=field,
                )
            if self.harmful_kind == "premature_terminal":
                if (
                    self.physical_owner_id is not None
                    or self.owner_resolution_interval is not None
                ):
                    _fail(
                        "state_bank.terminal_owner",
                        "premature terminal candidate must have null owner and owner interval",
                        field=field,
                    )
                if len(self.token_ids) != 1:
                    _fail(
                        "state_bank.terminal_token_count",
                        "premature terminal candidate score must contain exactly one token",
                        field=field,
                    )
                site = site_by_offset.get(0)
                if site is None or site.intended_token_type != "schema":
                    _fail(
                        "state_bank.terminal_intended_type",
                        "premature terminal boundary must be gated as object-row schema",
                        field=field,
                    )
            else:
                if (
                    self.physical_owner_id is None
                    or self.owner_resolution_interval is None
                ):
                    _fail(
                        "state_bank.owner_interval_missing",
                        "entity-eligible nonterminal candidate requires owner and resolution interval",
                        field=field,
                    )
                start, end = self.owner_resolution_interval
                if start != 0:
                    _fail(
                        "state_bank.owner_interval_start",
                        "owner-resolution interval must begin at candidate offset zero",
                        field=field,
                        start=start,
                    )
                missing_sites = [
                    offset
                    for offset in range(start, end)
                    if offset not in site_by_offset
                ]
                if missing_sites:
                    _fail(
                        "state_bank.selected_site_missing",
                        "every entity score token requires an intended token-type site",
                        field=field,
                        missing_offsets=missing_sites,
                    )
                if self.geometry_review_status != "trusted":
                    coordinate_offsets = [
                        site.candidate_token_offset
                        for site in self.selected_sites
                        if (
                            start <= site.candidate_token_offset < end
                            and site.intended_token_type == "coordinate"
                        )
                    ]
                    if coordinate_offsets:
                        _fail(
                            "state_bank.entity_transition_geometry_untrusted",
                            "entity-transition candidate resolving through coordinate tokens requires trusted geometry review",
                            field=field,
                            coordinate_offsets=coordinate_offsets,
                        )
            if self.role == "positive" and self.coverage_status != "uncovered":
                _fail(
                    "state_bank.positive_coverage",
                    "entity-transition positive must be explicitly uncovered",
                    field=field,
                )
            if self.harmful_kind == "duplicate" and self.coverage_status != "covered":
                _fail(
                    "state_bank.duplicate_coverage",
                    "duplicate harmful candidate must be explicitly covered",
                    field=field,
                )
        if self.geometry_eligible:
            if self.coordinate_decision is None:
                _fail(
                    "state_bank.coordinate_decision_missing",
                    "geometry-eligible candidate requires reviewed coordinate decision",
                    field=field,
                )
            decision = self.coordinate_decision
            assert decision is not None
            if decision.candidate_token_offset >= len(self.token_ids):
                _fail(
                    "state_bank.coordinate_offset_bounds",
                    "coordinate decision offset exceeds candidate token count",
                    field=field,
                    offset=decision.candidate_token_offset,
                    token_count=len(self.token_ids),
                )
            site = site_by_offset.get(decision.candidate_token_offset)
            if site is None or site.intended_token_type != "coordinate":
                _fail(
                    "state_bank.coordinate_intended_type",
                    "first-wrong-coordinate site must be gated as coordinate",
                    field=field,
                )
        elif self.coordinate_decision is not None:
            _fail(
                "state_bank.coordinate_ineligible_metadata",
                "coordinate decision may appear only on a geometry-eligible candidate",
                field=field,
            )
        expected_selected_offsets: set[int] = set()
        if self.entity_eligible:
            if self.harmful_kind == "premature_terminal":
                expected_selected_offsets.add(0)
            elif self.owner_resolution_interval is not None:
                start, end = self.owner_resolution_interval
                expected_selected_offsets.update(range(start, end))
        if self.geometry_eligible and self.coordinate_decision is not None:
            expected_selected_offsets.add(
                self.coordinate_decision.candidate_token_offset
            )
        actual_selected_offsets = {
            site.candidate_token_offset for site in self.selected_sites
        }
        if actual_selected_offsets != expected_selected_offsets:
            _fail(
                "state_bank.selected_site_scope",
                "selected sites must equal the union of enabled objective positions",
                field=field,
                expected_offsets=sorted(expected_selected_offsets),
                actual_offsets=sorted(actual_selected_offsets),
            )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "role": self.role,
            "harmful_kind": self.harmful_kind,
            "token_ids": list(self.token_ids),
            "token_ids_sha256": self.token_ids_sha256,
            "physical_owner_id": self.physical_owner_id,
            "coverage_status": self.coverage_status,
            "entity_review_status": self.entity_review_status,
            "geometry_review_status": self.geometry_review_status,
            "entity_eligible": self.entity_eligible,
            "geometry_eligible": self.geometry_eligible,
            "owner_resolution_interval": (
                None
                if self.owner_resolution_interval is None
                else list(self.owner_resolution_interval)
            ),
            "coordinate_decision": (
                None
                if self.coordinate_decision is None
                else self.coordinate_decision.to_artifact_dict()
            ),
            "selected_sites": [site.to_artifact_dict() for site in self.selected_sites],
            "generation_provenance": self.generation_provenance.to_artifact_dict(),
            "evidence_text": self.evidence_text,
        }


@dataclass(frozen=True)
class StateBankEvent:
    event_id: str
    image: ImageIdentity
    split: str
    split_group_id: str
    executed_prompt_token_ids: tuple[int, ...]
    executed_prompt_token_ids_sha256: str
    image_pad_interval: tuple[int, int]
    prefix_token_ids: tuple[int, ...]
    prefix_token_ids_sha256: str
    physical_entities: tuple[PhysicalEntity, ...]
    prefix_object_row_count: int
    prefix_coverage_status: str
    prefix_covered_owner_proofs: tuple[PrefixCoveredOwnerProof, ...]
    entity_transition_eligible: bool
    coordinate_boundary_eligible: bool
    candidates: tuple[StateBankCandidate, ...]
    review_provenance: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, field: str = "event"
    ) -> "StateBankEvent":
        checked = _mapping(value, field=field)
        _require_exact_keys(
            checked,
            {
                "event_id",
                "image",
                "split",
                "split_group_id",
                "executed_prompt_token_ids",
                "executed_prompt_token_ids_sha256",
                "image_pad_interval",
                "prefix_token_ids",
                "prefix_token_ids_sha256",
                "physical_entities",
                "prefix_object_row_count",
                "prefix_coverage_status",
                "prefix_covered_owner_proofs",
                "entity_transition_eligible",
                "coordinate_boundary_eligible",
                "candidates",
                "review_provenance",
            },
            field=field,
        )
        prompt_ids = _token_ids(
            checked["executed_prompt_token_ids"],
            field=f"{field}.executed_prompt_token_ids",
        )
        prompt_hash = _string(
            checked["executed_prompt_token_ids_sha256"],
            field=f"{field}.executed_prompt_token_ids_sha256",
        )
        _validate_token_hash(
            prompt_ids, prompt_hash, field=f"{field}.executed_prompt_token_ids"
        )
        prefix_ids = _token_ids(
            checked["prefix_token_ids"],
            field=f"{field}.prefix_token_ids",
            allow_empty=True,
        )
        prefix_hash = _string(
            checked["prefix_token_ids_sha256"], field=f"{field}.prefix_token_ids_sha256"
        )
        _validate_token_hash(prefix_ids, prefix_hash, field=f"{field}.prefix_token_ids")
        image = ImageIdentity.from_mapping(
            _mapping(checked["image"], field=f"{field}.image")
        )
        split = _choice(checked["split"], _SPLITS, field=f"{field}.split")
        split_group_id = _string(
            checked["split_group_id"], field=f"{field}.split_group_id"
        )
        expected_group = f"image:{image.image_id}"
        if split_group_id != expected_group:
            _fail(
                "state_bank.split_group",
                "split group must be derived from physical image identity",
                field=field,
                expected=expected_group,
                actual=split_group_id,
            )
        image_pad_interval = _interval(
            checked["image_pad_interval"],
            field=f"{field}.image_pad_interval",
            upper_bound=len(prompt_ids),
        )
        entities = tuple(
            PhysicalEntity.from_mapping(
                item, field=f"{field}.physical_entities[{index}]"
            )
            for index, item in enumerate(
                _sequence(
                    checked["physical_entities"], field=f"{field}.physical_entities"
                )
            )
        )
        if not entities:
            _fail(
                "state_bank.entity_ledger_empty",
                "admitted image requires a physical-entity ledger",
                field=field,
            )
        entity_ids = [entity.entity_id for entity in entities]
        _require_unique(
            entity_ids,
            code="state_bank.entity_id_duplicate",
            field=f"{field}.physical_entities",
        )
        prefix_owner_proofs = tuple(
            PrefixCoveredOwnerProof.from_mapping(
                item, field=f"{field}.prefix_covered_owner_proofs[{index}]"
            )
            for index, item in enumerate(
                _sequence(
                    checked["prefix_covered_owner_proofs"],
                    field=f"{field}.prefix_covered_owner_proofs",
                )
            )
        )
        proof_indices = tuple(
            item.prefix_object_row_index for item in prefix_owner_proofs
        )
        prefix_object_row_count = _require_nonnegative_int(
            checked["prefix_object_row_count"],
            field=f"{field}.prefix_object_row_count",
        )
        prefix_coverage_status = _choice(
            checked["prefix_coverage_status"],
            _PREFIX_COVERAGE_STATUSES,
            field=f"{field}.prefix_coverage_status",
        )
        if proof_indices != tuple(range(len(prefix_owner_proofs))):
            _fail(
                "state_bank.prefix_owner_proof_rows",
                "prefix owner proofs must cover every prior object row in order",
                field=field,
                row_indices=list(proof_indices),
            )
        if prefix_coverage_status == "empty":
            if prefix_object_row_count != 0 or prefix_ids or prefix_owner_proofs:
                _fail(
                    "state_bank.prefix_coverage_empty",
                    "empty prefix coverage requires zero object rows, empty prefix tokens, and zero owner proofs",
                    field=field,
                    prefix_object_row_count=prefix_object_row_count,
                    prefix_token_count=len(prefix_ids),
                    proof_count=len(prefix_owner_proofs),
                )
        elif prefix_coverage_status == "resolved":
            if (
                prefix_object_row_count == 0
                or not prefix_ids
                or len(prefix_owner_proofs) != prefix_object_row_count
                or proof_indices != tuple(range(prefix_object_row_count))
            ):
                _fail(
                    "state_bank.prefix_coverage_resolved",
                    "resolved prefix coverage requires a nonempty prefix and one ordered owner proof for every prior object row",
                    field=field,
                    prefix_object_row_count=prefix_object_row_count,
                    prefix_token_count=len(prefix_ids),
                    proof_count=len(prefix_owner_proofs),
                    proof_row_indices=list(proof_indices),
                )
        else:
            if (
                prefix_object_row_count == 0
                or not prefix_ids
                or prefix_owner_proofs
            ):
                _fail(
                    "state_bank.prefix_coverage_unresolved",
                    "unresolved prefix coverage requires one or more prior object rows, a nonempty prefix, and zero owner proofs",
                    field=field,
                    prefix_object_row_count=prefix_object_row_count,
                    prefix_token_count=len(prefix_ids),
                    proof_count=len(prefix_owner_proofs),
                )
        candidates = tuple(
            StateBankCandidate.from_mapping(item, field=f"{field}.candidates[{index}]")
            for index, item in enumerate(
                _sequence(checked["candidates"], field=f"{field}.candidates")
            )
        )
        if not candidates:
            _fail(
                "state_bank.candidates_empty",
                "event requires at least one candidate",
                field=field,
            )
        _require_unique(
            [candidate.candidate_id for candidate in candidates],
            code="state_bank.candidate_id_duplicate",
            field=f"{field}.candidates",
        )
        event = cls(
            event_id=_string(checked["event_id"], field=f"{field}.event_id"),
            image=image,
            split=split,
            split_group_id=split_group_id,
            executed_prompt_token_ids=prompt_ids,
            executed_prompt_token_ids_sha256=prompt_hash,
            image_pad_interval=image_pad_interval,
            prefix_token_ids=prefix_ids,
            prefix_token_ids_sha256=prefix_hash,
            physical_entities=entities,
            prefix_object_row_count=prefix_object_row_count,
            prefix_coverage_status=prefix_coverage_status,
            prefix_covered_owner_proofs=prefix_owner_proofs,
            entity_transition_eligible=_bool(
                checked["entity_transition_eligible"],
                field=f"{field}.entity_transition_eligible",
            ),
            coordinate_boundary_eligible=_bool(
                checked["coordinate_boundary_eligible"],
                field=f"{field}.coordinate_boundary_eligible",
            ),
            candidates=candidates,
            review_provenance=freeze_json(
                _mapping(
                    checked["review_provenance"], field=f"{field}.review_provenance"
                )
            ),
        )
        event._validate_semantics(field=field)
        return event

    def _validate_semantics(self, *, field: str) -> None:
        entity_by_id = {entity.entity_id: entity for entity in self.physical_entities}
        covered_owner_ids = frozenset(
            proof.owner_id for proof in self.prefix_covered_owner_proofs
        )
        for proof in self.prefix_covered_owner_proofs:
            owner = entity_by_id.get(proof.owner_id)
            if owner is None or not owner.entity_trusted:
                _fail(
                    "state_bank.prefix_owner_untrusted",
                    "prefix-covered owner proof requires a trusted physical ledger owner",
                    event_id=self.event_id,
                    owner_id=proof.owner_id,
                )
        for candidate in self.candidates:
            provenance = candidate.generation_provenance
            if (
                provenance.prompt_token_ids_sha256
                != self.executed_prompt_token_ids_sha256
            ):
                _fail(
                    "state_bank.candidate_prompt_provenance",
                    "candidate generation provenance does not bind the exact executed prompt",
                    event_id=self.event_id,
                    candidate_id=candidate.candidate_id,
                )
            if provenance.prefix_token_ids_sha256 != self.prefix_token_ids_sha256:
                _fail(
                    "state_bank.candidate_prefix_provenance",
                    "candidate generation provenance does not bind the exact prefix",
                    event_id=self.event_id,
                    candidate_id=candidate.candidate_id,
                )
            if (
                candidate.physical_owner_id is not None
                and candidate.physical_owner_id not in entity_by_id
            ):
                _fail(
                    "state_bank.owner_missing",
                    "candidate physical owner is absent from the image ledger",
                    event_id=self.event_id,
                    candidate_id=candidate.candidate_id,
                    owner_id=candidate.physical_owner_id,
                )
            decision = candidate.coordinate_decision
            if decision is not None:
                owner = entity_by_id.get(decision.owner_id)
                if owner is None:
                    _fail(
                        "state_bank.geometry_owner_missing",
                        "coordinate decision owner is absent from the image ledger",
                        event_id=self.event_id,
                        candidate_id=candidate.candidate_id,
                        owner_id=decision.owner_id,
                    )
                if owner is not None and not owner.geometry_trusted:
                    _fail(
                        "state_bank.geometry_owner_untrusted",
                        "geometry-eligible decision requires trusted ledger geometry",
                        event_id=self.event_id,
                        candidate_id=candidate.candidate_id,
                        owner_id=decision.owner_id,
                    )
                if candidate.physical_owner_id != decision.owner_id:
                    _fail(
                        "state_bank.geometry_same_owner",
                        "coordinate correction owner must equal the trusted candidate owner",
                        event_id=self.event_id,
                        candidate_id=candidate.candidate_id,
                        candidate_owner_id=candidate.physical_owner_id,
                        correction_owner_id=decision.owner_id,
                    )
                for observation in decision.observations:
                    if observation.candidate_token_offset >= len(candidate.token_ids):
                        _fail(
                            "state_bank.coordinate_offset_bounds",
                            "coordinate observation offset exceeds candidate token count",
                            event_id=self.event_id,
                            candidate_id=candidate.candidate_id,
                            offset=observation.candidate_token_offset,
                        )
            if candidate.entity_eligible and candidate.physical_owner_id is not None:
                owner = entity_by_id[candidate.physical_owner_id]
                if not owner.entity_trusted:
                    _fail(
                        "state_bank.entity_owner_untrusted",
                        "entity-eligible owner must be trusted in the physical ledger",
                        event_id=self.event_id,
                        candidate_id=candidate.candidate_id,
                        owner_id=candidate.physical_owner_id,
                    )
                if (
                    candidate.role == "positive"
                    and candidate.physical_owner_id in covered_owner_ids
                ):
                    _fail(
                        "state_bank.positive_prefix_covered",
                        "transition positive owner must be absent from the exact-prefix covered set",
                        event_id=self.event_id,
                        candidate_id=candidate.candidate_id,
                        owner_id=candidate.physical_owner_id,
                    )
                if (
                    candidate.harmful_kind == "duplicate"
                    and candidate.physical_owner_id not in covered_owner_ids
                ):
                    _fail(
                        "state_bank.duplicate_prefix_uncovered",
                        "physical-duplicate harmful owner must be present in the exact-prefix covered set",
                        event_id=self.event_id,
                        candidate_id=candidate.candidate_id,
                        owner_id=candidate.physical_owner_id,
                    )
        entity_candidates = [
            candidate for candidate in self.candidates if candidate.entity_eligible
        ]
        geometry_candidates = [
            candidate for candidate in self.candidates if candidate.geometry_eligible
        ]
        harmful_candidates = [
            candidate for candidate in self.candidates if candidate.role == "harmful"
        ]
        greedy_candidates = [
            candidate
            for candidate in self.candidates
            if candidate.generation_provenance.mode == "greedy"
        ]
        if self.prefix_coverage_status == "unresolved":
            if self.entity_transition_eligible:
                _fail(
                    "state_bank.unresolved_prefix_entity_event",
                    "unresolved prefix coverage cannot support entity-transition supervision",
                    event_id=self.event_id,
                )
            entity_eligible_ids = [
                candidate.candidate_id
                for candidate in self.candidates
                if candidate.entity_eligible
            ]
            if entity_eligible_ids:
                _fail(
                    "state_bank.unresolved_prefix_entity_candidate",
                    "every candidate must be entity-ineligible when prefix coverage is unresolved",
                    event_id=self.event_id,
                    candidate_ids=entity_eligible_ids,
                )
            coordinate_candidates_with_known_coverage = [
                candidate.candidate_id
                for candidate in geometry_candidates
                if candidate.coverage_status != "unknown"
            ]
            if coordinate_candidates_with_known_coverage:
                _fail(
                    "state_bank.unresolved_prefix_coordinate_coverage",
                    "coordinate candidates must declare unknown coverage when prefix coverage is unresolved",
                    event_id=self.event_id,
                    candidate_ids=coordinate_candidates_with_known_coverage,
                )
        if (
            self.entity_transition_eligible
            and self.prefix_coverage_status != "resolved"
        ):
            _fail(
                "state_bank.entity_prefix_coverage_not_resolved",
                "entity-transition supervision requires fully resolved prefix coverage",
                event_id=self.event_id,
                prefix_coverage_status=self.prefix_coverage_status,
            )
        if self.entity_transition_eligible:
            if (
                len(harmful_candidates) != 1
                or len(greedy_candidates) != 1
                or greedy_candidates[0] is not harmful_candidates[0]
            ):
                _fail(
                    "state_bank.greedy_harmful_identity",
                    "the sole harmful branch must be the sole producer-declared greedy candidate",
                    event_id=self.event_id,
                    greedy_candidate_ids=[item.candidate_id for item in greedy_candidates],
                    harmful_candidate_ids=[
                        item.candidate_id for item in harmful_candidates
                    ],
                )
        elif self.coordinate_boundary_eligible:
            # A coordinate-only event is a diagnostic replay of one exact
            # greedy row, not an entity-transition preference.  Its sole
            # greedy candidate is therefore intentionally diagnostic and has
            # no harmful branch.  The coordinate objective still consumes its
            # reviewed wrong-boundary decision below.
            if harmful_candidates:
                _fail(
                    "state_bank.coordinate_harmful_candidate",
                    "coordinate-only event must not declare a harmful candidate",
                    event_id=self.event_id,
                    harmful_candidate_ids=[
                        item.candidate_id for item in harmful_candidates
                    ],
                )
            if len(greedy_candidates) != 1:
                _fail(
                    "state_bank.coordinate_greedy_diagnostic",
                    "coordinate-only event requires exactly one diagnostic greedy candidate",
                    event_id=self.event_id,
                    greedy_candidate_ids=[item.candidate_id for item in greedy_candidates],
                    greedy_roles=[item.role for item in greedy_candidates],
                )
            diagnostic = greedy_candidates[0]
            if not (
                diagnostic.role == "diagnostic"
                and diagnostic.harmful_kind is None
                and not diagnostic.entity_eligible
                and diagnostic.geometry_eligible
                and diagnostic.coordinate_decision is not None
            ):
                _fail(
                    "state_bank.coordinate_greedy_diagnostic",
                    "coordinate-only greedy candidate must be diagnostic, geometry-eligible, entity-ineligible, and carry a coordinate decision",
                    event_id=self.event_id,
                    candidate_id=diagnostic.candidate_id,
                    role=diagnostic.role,
                    harmful_kind=diagnostic.harmful_kind,
                    entity_eligible=diagnostic.entity_eligible,
                    geometry_eligible=diagnostic.geometry_eligible,
                    has_coordinate_decision=diagnostic.coordinate_decision is not None,
                )
        non_sampled_positives = [
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.role == "positive"
            and candidate.generation_provenance.mode != "sampled"
        ]
        if non_sampled_positives:
            _fail(
                "state_bank.positive_not_sampled",
                "every positive must have same-prefix sampled provenance",
                event_id=self.event_id,
                candidate_ids=non_sampled_positives,
            )
        if self.entity_transition_eligible:
            positives = [
                candidate
                for candidate in entity_candidates
                if candidate.role == "positive"
            ]
            harmful = [
                candidate
                for candidate in entity_candidates
                if candidate.role == "harmful"
            ]
            if not positives or len(harmful) != 1:
                _fail(
                    "state_bank.entity_event_group",
                    "entity event requires one or more positives and exactly one harmful branch",
                    event_id=self.event_id,
                    positive_count=len(positives),
                    harmful_count=len(harmful),
                )
        elif entity_candidates:
            _fail(
                "state_bank.entity_event_disabled",
                "event-level entity eligibility conflicts with candidate declarations",
                event_id=self.event_id,
            )
        if self.coordinate_boundary_eligible and len(geometry_candidates) != 1:
            _fail(
                "state_bank.geometry_event_group",
                "coordinate event requires exactly one geometry-eligible wrong decision",
                event_id=self.event_id,
                geometry_candidate_count=len(geometry_candidates),
            )
        if not self.coordinate_boundary_eligible and geometry_candidates:
            _fail(
                "state_bank.geometry_event_disabled",
                "event-level coordinate eligibility conflicts with candidate declarations",
                event_id=self.event_id,
            )

    def to_raw_example(self, *, example_id: str) -> RawExample:
        image = ImageRef(
            declared_path=str(self.image.path),
            path=self.image.path,
            width=self.image.width,
            height=self.image.height,
            stat=image_stat_fingerprint(self.image.path),
        )
        objects = tuple(
            RawObject(
                object_id=entity.entity_id,
                description=entity.category,
                bbox=entity.reference_bbox,
                metadata=freeze_json(
                    {
                        "entity_trusted": entity.entity_trusted,
                        "geometry_trusted": entity.geometry_trusted,
                        "review_source": entity.review_source,
                    }
                ),
            )
            for entity in self.physical_entities
        )
        return RawExample(
            example_id=example_id,
            image=image,
            objects=objects,
            metadata=freeze_json(
                {
                    "rollout_calibration_event_id": self.event_id,
                    "image_id": self.image.image_id,
                }
            ),
            source=SourceProvenance(
                source_path=self.image.path,
                row_number=1,
                row_sha256=self.image.content_sha256,
                source_format=STATE_BANK_SCHEMA_VERSION,
            ),
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "image": self.image.to_artifact_dict(),
            "split": self.split,
            "split_group_id": self.split_group_id,
            "executed_prompt_token_ids": list(self.executed_prompt_token_ids),
            "executed_prompt_token_ids_sha256": self.executed_prompt_token_ids_sha256,
            "image_pad_interval": list(self.image_pad_interval),
            "prefix_token_ids": list(self.prefix_token_ids),
            "prefix_token_ids_sha256": self.prefix_token_ids_sha256,
            "physical_entities": [
                entity.to_artifact_dict() for entity in self.physical_entities
            ],
            "prefix_object_row_count": self.prefix_object_row_count,
            "prefix_coverage_status": self.prefix_coverage_status,
            "prefix_covered_owner_proofs": [
                proof.to_artifact_dict() for proof in self.prefix_covered_owner_proofs
            ],
            "entity_transition_eligible": self.entity_transition_eligible,
            "coordinate_boundary_eligible": self.coordinate_boundary_eligible,
            "candidates": [
                candidate.to_artifact_dict() for candidate in self.candidates
            ],
            "review_provenance": _thaw(self.review_provenance),
        }


@dataclass(frozen=True)
class StateBankManifest:
    bank_id: str
    source_checkpoint_id: str
    records_sha256: str
    record_count: int
    source_checkpoint: CheckpointIdentity
    prompt_identity_sha256: str
    split_assignments: tuple[dict[str, Any], ...]
    split_counts: Mapping[str, int]
    event_family_counts: Mapping[str, int]
    source_artifacts: tuple[dict[str, str], ...]
    rejection_reasons: Mapping[str, int]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StateBankManifest":
        checked = _mapping(value, field="manifest")
        _require_exact_keys(
            checked,
            {
                "schema_version",
                "bank_id",
                "source_checkpoint_id",
                "records_file",
                "records_sha256",
                "record_count",
                "source_checkpoint",
                "prompt_identity_sha256",
                "split_assignments",
                "split_counts",
                "event_family_counts",
                "source_artifacts",
                "blind_image_ids",
                "rejection_reasons",
            },
            field="manifest",
        )
        if checked["schema_version"] != STATE_BANK_SCHEMA_VERSION:
            _fail(
                "state_bank.schema_version",
                "unsupported rollout state-bank schema version",
                actual=checked["schema_version"],
                expected=STATE_BANK_SCHEMA_VERSION,
            )
        if checked["records_file"] != STATE_BANK_RECORDS_NAME:
            _fail(
                "state_bank.records_file",
                "state-bank records file must use the canonical basename",
                actual=checked["records_file"],
            )
        blind_ids = tuple(
            _image_id(item, field=f"manifest.blind_image_ids[{index}]")
            for index, item in enumerate(
                _sequence(checked["blind_image_ids"], field="manifest.blind_image_ids")
            )
        )
        if blind_ids != tuple(sorted(BLIND_IMAGE_IDS)):
            _fail(
                "state_bank.blind_cohort_contract",
                "manifest must bind the complete pilot blind cohort",
                expected=list(sorted(BLIND_IMAGE_IDS)),
                actual=list(blind_ids),
            )
        assignments = tuple(
            _split_assignment(item, field=f"manifest.split_assignments[{index}]")
            for index, item in enumerate(
                _sequence(
                    checked["split_assignments"], field="manifest.split_assignments"
                )
            )
        )
        artifacts = tuple(
            _source_artifact(item, field=f"manifest.source_artifacts[{index}]")
            for index, item in enumerate(
                _sequence(
                    checked["source_artifacts"], field="manifest.source_artifacts"
                )
            )
        )
        rejections = _count_mapping(
            checked["rejection_reasons"], field="manifest.rejection_reasons"
        )
        split_counts = _count_mapping(
            checked["split_counts"], field="manifest.split_counts"
        )
        event_family_counts = _count_mapping(
            checked["event_family_counts"], field="manifest.event_family_counts"
        )
        manifest = cls(
            bank_id=_string(checked["bank_id"], field="manifest.bank_id"),
            source_checkpoint_id=_string(
                checked["source_checkpoint_id"], field="manifest.source_checkpoint_id"
            ),
            records_sha256=_string(
                checked["records_sha256"], field="manifest.records_sha256"
            ),
            record_count=_require_positive_int(
                checked["record_count"], field="manifest.record_count"
            ),
            source_checkpoint=CheckpointIdentity.from_mapping(
                _mapping(
                    checked["source_checkpoint"], field="manifest.source_checkpoint"
                )
            ),
            prompt_identity_sha256=_string(
                checked["prompt_identity_sha256"],
                field="manifest.prompt_identity_sha256",
            ),
            split_assignments=assignments,
            split_counts=freeze_json(split_counts),
            event_family_counts=freeze_json(event_family_counts),
            source_artifacts=artifacts,
            rejection_reasons=freeze_json(rejections),
        )
        _require_sha256(manifest.bank_id, field="manifest.bank_id")
        _require_sha256(
            manifest.source_checkpoint_id, field="manifest.source_checkpoint_id"
        )
        _require_sha256(manifest.records_sha256, field="manifest.records_sha256")
        _require_sha256(
            manifest.prompt_identity_sha256, field="manifest.prompt_identity_sha256"
        )
        expected_source_checkpoint_id = sha256_json(
            manifest.source_checkpoint.to_artifact_dict()
        )
        if manifest.source_checkpoint_id != expected_source_checkpoint_id:
            _fail(
                "state_bank.source_checkpoint_id",
                "source checkpoint id does not match its composite identity",
                expected=expected_source_checkpoint_id,
                actual=manifest.source_checkpoint_id,
            )
        expected_bank_id = sha256_json(manifest.identity_determinants())
        if manifest.bank_id != expected_bank_id:
            _fail(
                "state_bank.bank_id",
                "state-bank identity does not match canonical manifest determinants",
                expected=expected_bank_id,
                actual=manifest.bank_id,
            )
        return manifest

    def identity_determinants(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_BANK_SCHEMA_VERSION,
            "records_file": STATE_BANK_RECORDS_NAME,
            "records_sha256": self.records_sha256,
            "record_count": self.record_count,
            "source_checkpoint_id": self.source_checkpoint_id,
            "source_checkpoint": self.source_checkpoint.to_artifact_dict(),
            "prompt_identity_sha256": self.prompt_identity_sha256,
            "split_assignments": list(self.split_assignments),
            "split_counts": _thaw(self.split_counts),
            "event_family_counts": _thaw(self.event_family_counts),
            "source_artifacts": list(self.source_artifacts),
            "blind_image_ids": list(sorted(BLIND_IMAGE_IDS)),
            "rejection_reasons": _thaw(self.rejection_reasons),
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {"bank_id": self.bank_id, **self.identity_determinants()}


@dataclass(frozen=True)
class StateBankValidationReceipt:
    bank_id: str
    source_checkpoint_id: str
    source_checkpoint: CheckpointIdentity
    records_sha256: str
    record_count: int
    split_counts: Mapping[str, int]
    event_family_counts: Mapping[str, int]
    rejection_reasons: Mapping[str, int]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "source_checkpoint_id": self.source_checkpoint_id,
            "source_checkpoint": self.source_checkpoint.to_artifact_dict(),
            "records_sha256": self.records_sha256,
            "record_count": self.record_count,
            "split_counts": _thaw(self.split_counts),
            "event_family_counts": _thaw(self.event_family_counts),
            "rejection_reasons": _thaw(self.rejection_reasons),
            "status": "validated",
        }


@dataclass(frozen=True)
class LoadedStateBank:
    root: Path
    manifest: StateBankManifest
    records: tuple[StateBankEvent, ...]
    validation_receipt: StateBankValidationReceipt

    def records_for_split(self, split: str) -> tuple[StateBankEvent, ...]:
        checked = _choice(split, _SPLITS, field="split")
        return tuple(record for record in self.records if record.split == checked)


@dataclass(frozen=True)
class StateBankManifestBinding:
    """Manifest-only checkpoint/config binding available before model loading."""

    source_checkpoint_id: str
    bank_id: str
    source_composite_fingerprint: str
    source_checkpoint: CheckpointIdentity
    prompt_identity_sha256: str
    records_sha256: str
    record_count: int
    split_counts: Mapping[str, int]
    event_family_counts: Mapping[str, int]

    @property
    def fingerprint(self) -> str:
        return self.bank_id

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "source_checkpoint_id": self.source_checkpoint_id,
            "bank_id": self.bank_id,
            "fingerprint": self.fingerprint,
            "source_composite_fingerprint": self.source_composite_fingerprint,
            "source_checkpoint": self.source_checkpoint.to_artifact_dict(),
            "prompt_identity_sha256": self.prompt_identity_sha256,
            "records_sha256": self.records_sha256,
            "record_count": self.record_count,
            "split_counts": _thaw(self.split_counts),
            "event_family_counts": _thaw(self.event_family_counts),
        }


def assemble_state_bank(
    *,
    output_dir: str | Path,
    rollout_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    source_checkpoint: CheckpointIdentity | Mapping[str, Any],
    prompt_identity_sha256: str,
    source_artifacts: Sequence[Mapping[str, Any]],
) -> StateBankManifest:
    """Join exact rollout evidence to explicit review decisions and write once."""

    checkpoint = _checkpoint_identity(source_checkpoint)
    _require_sha256(prompt_identity_sha256, field="prompt_identity_sha256")
    rollouts = tuple(
        _mapping(row, field=f"rollout_rows[{index}]")
        for index, row in enumerate(rollout_rows)
    )
    reviews = tuple(
        _mapping(row, field=f"review_rows[{index}]")
        for index, row in enumerate(review_rows)
    )
    if not rollouts:
        _fail("state_bank.assembler_rollouts_empty", "assembler requires rollout rows")
    _reject_blind_sources(rollouts, reviews)
    rollout_by_id = _unique_rows_by_id(rollouts, field="event_id", owner="rollout_rows")
    review_by_id = _unique_rows_by_id(reviews, field="event_id", owner="review_rows")
    if set(rollout_by_id) != set(review_by_id):
        _fail(
            "state_bank.assembler_join",
            "rollout and review event identifiers must match exactly",
            rollout_only=sorted(set(rollout_by_id) - set(review_by_id)),
            review_only=sorted(set(review_by_id) - set(rollout_by_id)),
        )
    records: list[StateBankEvent] = []
    rejection_reasons: Counter[str] = Counter()
    for event_id in sorted(rollout_by_id):
        review = review_by_id[event_id]
        admission = _choice(
            review.get("admission_status"),
            frozenset({"accepted", "rejected"}),
            field=f"review_rows[{event_id}].admission_status",
        )
        if admission == "rejected":
            reason = _string(
                review.get("rejection_reason"),
                field=f"review_rows[{event_id}].rejection_reason",
            )
            rejection_reasons[reason] += 1
            continue
        if review.get("rejection_reason") is not None:
            _fail(
                "state_bank.assembler_accepted_rejection_reason",
                "accepted review row must not declare rejection_reason",
                event_id=event_id,
            )
        joined = _join_rollout_and_review(rollout_by_id[event_id], review)
        records.append(StateBankEvent.from_mapping(joined, field=f"events[{event_id}]"))
    if not records:
        _fail(
            "state_bank.assembler_no_accepted_records",
            "assembler produced no accepted state-bank records",
        )
    records.sort(key=lambda item: item.event_id)
    _validate_record_collection(tuple(records))
    source_checkpoint_id = sha256_json(checkpoint.to_artifact_dict())
    _validate_record_checkpoint_provenance(
        tuple(records), source_checkpoint_id=source_checkpoint_id
    )
    return _write_state_bank(
        output_dir=Path(output_dir),
        records=tuple(records),
        source_checkpoint=checkpoint,
        prompt_identity_sha256=prompt_identity_sha256,
        source_artifacts=tuple(
            _source_artifact(item, field=f"source_artifacts[{index}]")
            for index, item in enumerate(source_artifacts)
        ),
        rejection_reasons=dict(sorted(rejection_reasons.items())),
    )


def load_state_bank(
    manifest_path: str | Path,
    *,
    expected_source_checkpoint: CheckpointIdentity | Mapping[str, Any],
    expected_prompt_identity_sha256: str,
) -> LoadedStateBank:
    path = Path(manifest_path).expanduser().resolve()
    payload = _read_json(path)
    manifest = StateBankManifest.from_mapping(payload)
    expected_checkpoint = _checkpoint_identity(expected_source_checkpoint)
    if manifest.source_checkpoint != expected_checkpoint:
        _fail(
            "state_bank.source_checkpoint_mismatch",
            "state bank belongs to a different source checkpoint",
            expected=expected_checkpoint.to_artifact_dict(),
            actual=manifest.source_checkpoint.to_artifact_dict(),
        )
    _require_sha256(
        expected_prompt_identity_sha256, field="expected_prompt_identity_sha256"
    )
    if manifest.prompt_identity_sha256 != expected_prompt_identity_sha256:
        _fail(
            "state_bank.prompt_identity_mismatch",
            "state bank prompt identity differs from configured prompt identity",
            expected=expected_prompt_identity_sha256,
            actual=manifest.prompt_identity_sha256,
        )
    records_path = path.parent / STATE_BANK_RECORDS_NAME
    if not records_path.is_file():
        _fail(
            "state_bank.records_missing",
            "state-bank records JSONL does not exist",
            path=str(records_path),
        )
    actual_records_sha = sha256_file(records_path)
    if actual_records_sha != manifest.records_sha256:
        _fail(
            "state_bank.records_hash",
            "state-bank records checksum does not match manifest",
            expected=manifest.records_sha256,
            actual=actual_records_sha,
        )
    records = _read_records(records_path)
    if len(records) != manifest.record_count:
        _fail(
            "state_bank.record_count",
            "state-bank record count does not match manifest",
            expected=manifest.record_count,
            actual=len(records),
        )
    _validate_record_collection(records)
    _validate_record_checkpoint_provenance(
        records, source_checkpoint_id=manifest.source_checkpoint_id
    )
    expected_assignments = _split_assignments(records)
    if tuple(manifest.split_assignments) != expected_assignments:
        _fail(
            "state_bank.split_manifest",
            "manifest split assignments do not match state-bank records",
            expected=list(expected_assignments),
            actual=list(manifest.split_assignments),
        )
    actual_split_counts, actual_event_family_counts = _record_counts(records)
    if dict(manifest.split_counts) != actual_split_counts:
        _fail(
            "state_bank.split_counts",
            "manifest split counts do not match loaded records",
            expected=actual_split_counts,
            actual=_thaw(manifest.split_counts),
        )
    if dict(manifest.event_family_counts) != actual_event_family_counts:
        _fail(
            "state_bank.event_family_counts",
            "manifest event-family counts do not match loaded records",
            expected=actual_event_family_counts,
            actual=_thaw(manifest.event_family_counts),
        )
    _validate_images(records)
    receipt = _validation_receipt(manifest, records)
    return LoadedStateBank(
        root=path.parent,
        manifest=manifest,
        records=records,
        validation_receipt=receipt,
    )


def load_state_bank_manifest_binding(
    manifest_path: str | Path,
) -> StateBankManifestBinding:
    """Strictly validate only the immutable manifest and expose config binding facts."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = StateBankManifest.from_mapping(_read_json(path))
    return StateBankManifestBinding(
        source_checkpoint_id=manifest.source_checkpoint_id,
        bank_id=manifest.bank_id,
        source_composite_fingerprint=manifest.source_checkpoint_id,
        source_checkpoint=manifest.source_checkpoint,
        prompt_identity_sha256=manifest.prompt_identity_sha256,
        records_sha256=manifest.records_sha256,
        record_count=manifest.record_count,
        split_counts=manifest.split_counts,
        event_family_counts=manifest.event_family_counts,
    )


def validate_state_bank_token_identity(
    bank: LoadedStateBank, token_identity: Any
) -> None:
    """Reject semantic token declarations that disagree with bound Qwen ids."""

    terminal_token_ids = tuple(int(value) for value in token_identity.im_end_token_ids)
    coordinate_token_ids = tuple(
        int(value) for value in token_identity.coordinate_token_ids
    )
    if len(terminal_token_ids) != 1 or len(coordinate_token_ids) != 1000:
        _fail(
            "state_bank.token_identity_shape",
            "bound Qwen token identity has unexpected terminal or coordinate shape",
            terminal_token_count=len(terminal_token_ids),
            coordinate_token_count=len(coordinate_token_ids),
        )
    coordinate_token_id_set = frozenset(coordinate_token_ids)
    for event in bank.records:
        for candidate in event.candidates:
            if candidate.harmful_kind == "premature_terminal" and (
                candidate.token_ids != terminal_token_ids
            ):
                _fail(
                    "state_bank.terminal_token_identity",
                    "premature-terminal candidate is not the bound Qwen terminal token",
                    event_id=event.event_id,
                    candidate_id=candidate.candidate_id,
                    expected_token_ids=list(terminal_token_ids),
                    actual_token_ids=list(candidate.token_ids),
                )
    for event in bank.records:
        for candidate in event.candidates:
            decision = candidate.coordinate_decision
            if decision is not None:
                for observation in decision.observations:
                    expected_coordinate_id = coordinate_token_ids[
                        observation.actual_coordinate_value
                    ]
                    actual_coordinate_id = candidate.token_ids[
                        observation.candidate_token_offset
                    ]
                    if actual_coordinate_id != expected_coordinate_id:
                        _fail(
                            "state_bank.coordinate_token_identity",
                            "coordinate observation value does not match its exact candidate token",
                            event_id=event.event_id,
                            candidate_id=candidate.candidate_id,
                            coordinate=observation.coordinate,
                            expected_token_id=expected_coordinate_id,
                            actual_token_id=actual_coordinate_id,
                        )
            if (
                candidate.entity_eligible
                and candidate.harmful_kind != "premature_terminal"
                and candidate.owner_resolution_interval is not None
                and candidate.geometry_review_status != "trusted"
            ):
                start, end = candidate.owner_resolution_interval
                actual_coordinate_offsets = [
                    offset
                    for offset in range(start, end)
                    if candidate.token_ids[offset] in coordinate_token_id_set
                ]
                if actual_coordinate_offsets:
                    _fail(
                        "state_bank.entity_transition_geometry_untrusted",
                        "entity-transition candidate resolving through bound coordinate tokens requires trusted geometry review",
                        event_id=event.event_id,
                        candidate_id=candidate.candidate_id,
                        coordinate_offsets=actual_coordinate_offsets,
                    )


def _join_rollout_and_review(
    rollout: Mapping[str, Any], review: Mapping[str, Any]
) -> dict[str, Any]:
    _require_exact_keys(
        rollout,
        {
            "event_id",
            "image",
            "split",
            "split_group_id",
            "executed_prompt_token_ids",
            "executed_prompt_token_ids_sha256",
            "image_pad_interval",
            "prefix_token_ids",
            "prefix_token_ids_sha256",
            "candidates",
        },
        field=f"rollout[{rollout.get('event_id', '?')}]",
    )
    _require_exact_keys(
        review,
        {
            "event_id",
            "admission_status",
            "rejection_reason",
            "physical_entities",
            "prefix_object_row_count",
            "prefix_coverage_status",
            "prefix_covered_owner_proofs",
            "entity_transition_eligible",
            "coordinate_boundary_eligible",
            "candidates",
            "review_provenance",
        },
        field=f"review[{review.get('event_id', '?')}]",
    )
    event_id = _string(rollout["event_id"], field="rollout.event_id")
    if review["event_id"] != event_id:
        _fail(
            "state_bank.assembler_event_id",
            "joined event identifiers differ",
            event_id=event_id,
        )
    rollout_candidates = _unique_rows_by_id(
        _sequence(rollout["candidates"], field=f"rollout[{event_id}].candidates"),
        field="candidate_id",
        owner=f"rollout[{event_id}].candidates",
    )
    review_candidates = _unique_rows_by_id(
        _sequence(review["candidates"], field=f"review[{event_id}].candidates"),
        field="candidate_id",
        owner=f"review[{event_id}].candidates",
    )
    if set(rollout_candidates) != set(review_candidates):
        _fail(
            "state_bank.assembler_candidate_join",
            "rollout and review candidate identifiers must match exactly",
            event_id=event_id,
            rollout_only=sorted(set(rollout_candidates) - set(review_candidates)),
            review_only=sorted(set(review_candidates) - set(rollout_candidates)),
        )
    joined_candidates: list[dict[str, Any]] = []
    rollout_candidate_keys = {
        "candidate_id",
        "token_ids",
        "token_ids_sha256",
        "generation_provenance",
        "evidence_text",
    }
    review_candidate_keys = {
        "candidate_id",
        "role",
        "harmful_kind",
        "physical_owner_id",
        "coverage_status",
        "entity_review_status",
        "geometry_review_status",
        "entity_eligible",
        "geometry_eligible",
        "owner_resolution_interval",
        "coordinate_decision",
        "selected_sites",
    }
    for candidate_id in sorted(rollout_candidates):
        rollout_candidate = rollout_candidates[candidate_id]
        review_candidate = review_candidates[candidate_id]
        _require_exact_keys(
            rollout_candidate,
            rollout_candidate_keys,
            field=f"rollout[{event_id}].candidate[{candidate_id}]",
        )
        _require_exact_keys(
            review_candidate,
            review_candidate_keys,
            field=f"review[{event_id}].candidate[{candidate_id}]",
        )
        joined_candidates.append({**dict(rollout_candidate), **dict(review_candidate)})
    return {
        "event_id": event_id,
        "image": rollout["image"],
        "split": rollout["split"],
        "split_group_id": rollout["split_group_id"],
        "executed_prompt_token_ids": rollout["executed_prompt_token_ids"],
        "executed_prompt_token_ids_sha256": rollout["executed_prompt_token_ids_sha256"],
        "image_pad_interval": rollout["image_pad_interval"],
        "prefix_token_ids": rollout["prefix_token_ids"],
        "prefix_token_ids_sha256": rollout["prefix_token_ids_sha256"],
        "physical_entities": review["physical_entities"],
        "prefix_object_row_count": review["prefix_object_row_count"],
        "prefix_coverage_status": review["prefix_coverage_status"],
        "prefix_covered_owner_proofs": review["prefix_covered_owner_proofs"],
        "entity_transition_eligible": review["entity_transition_eligible"],
        "coordinate_boundary_eligible": review["coordinate_boundary_eligible"],
        "candidates": joined_candidates,
        "review_provenance": review["review_provenance"],
    }


def _write_state_bank(
    *,
    output_dir: Path,
    records: tuple[StateBankEvent, ...],
    source_checkpoint: CheckpointIdentity,
    prompt_identity_sha256: str,
    source_artifacts: tuple[dict[str, str], ...],
    rejection_reasons: Mapping[str, int],
) -> StateBankManifest:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / STATE_BANK_RECORDS_NAME
    manifest_path = output_dir / STATE_BANK_MANIFEST_NAME
    if records_path.exists() or manifest_path.exists():
        _fail(
            "state_bank.immutable_output_exists",
            "immutable state-bank output already exists",
            output_dir=str(output_dir),
        )
    records_bytes = b"".join(
        (_canonical_json(record.to_artifact_dict()) + "\n").encode("utf-8")
        for record in records
    )
    records_sha = hashlib.sha256(records_bytes).hexdigest()
    split_counts, event_family_counts = _record_counts(records)
    determinants = {
        "schema_version": STATE_BANK_SCHEMA_VERSION,
        "records_file": STATE_BANK_RECORDS_NAME,
        "records_sha256": records_sha,
        "record_count": len(records),
        "source_checkpoint_id": sha256_json(source_checkpoint.to_artifact_dict()),
        "source_checkpoint": source_checkpoint.to_artifact_dict(),
        "prompt_identity_sha256": prompt_identity_sha256,
        "split_assignments": list(_split_assignments(records)),
        "split_counts": split_counts,
        "event_family_counts": event_family_counts,
        "source_artifacts": list(source_artifacts),
        "blind_image_ids": list(sorted(BLIND_IMAGE_IDS)),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }
    manifest_payload = {"bank_id": sha256_json(determinants), **determinants}
    _atomic_write_bytes(records_path, records_bytes)
    try:
        _atomic_write_bytes(
            manifest_path,
            (json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
    except BaseException:
        records_path.unlink(missing_ok=True)
        raise
    return StateBankManifest.from_mapping(manifest_payload)


def _read_records(path: Path) -> tuple[StateBankEvent, ...]:
    records: list[StateBankEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                _fail(
                    "state_bank.blank_record",
                    "state-bank JSONL contains a blank row",
                    row_number=row_number,
                )
            try:
                payload = _strict_json_loads(raw_line)
            except (json.JSONDecodeError, ValueError) as exc:
                _fail(
                    "state_bank.record_json",
                    "state-bank record is not valid JSON",
                    row_number=row_number,
                    error=str(exc),
                )
            records.append(
                StateBankEvent.from_mapping(payload, field=f"records[{row_number}]")
            )
    return tuple(records)


def _validate_record_collection(records: tuple[StateBankEvent, ...]) -> None:
    _require_unique(
        [record.event_id for record in records],
        code="state_bank.event_id_duplicate",
        field="records",
    )
    assignment_by_image: dict[int, tuple[str, str, str]] = {}
    assignment_by_content: dict[str, tuple[str, str]] = {}
    train_counts: Counter[int] = Counter()
    for record in records:
        _reject_blind_image_id(
            record.image.image_id, field=f"event[{record.event_id}].image_id"
        )
        assignment = (record.split, record.split_group_id, record.image.content_sha256)
        previous = assignment_by_image.setdefault(record.image.image_id, assignment)
        if previous != assignment:
            _fail(
                "state_bank.image_split_leakage",
                "one physical image identity appears in conflicting split assignments",
                image_id=record.image.image_id,
                first=list(previous),
                later=list(assignment),
            )
        content_assignment = (record.split, record.split_group_id)
        prior_content = assignment_by_content.setdefault(
            record.image.content_sha256, content_assignment
        )
        if prior_content != content_assignment:
            _fail(
                "state_bank.image_content_split_leakage",
                "identical image bytes appear in conflicting split assignments",
                content_sha256=record.image.content_sha256,
            )
        if record.split == "train":
            train_counts[record.image.image_id] += 1
    excessive = {
        image_id: count for image_id, count in train_counts.items() if count > 4
    }
    if excessive:
        _fail(
            "state_bank.max_train_states_per_image",
            "pilot permits at most four training states per image",
            counts=excessive,
        )


def _validate_record_checkpoint_provenance(
    records: tuple[StateBankEvent, ...], *, source_checkpoint_id: str
) -> None:
    for record in records:
        for candidate in record.candidates:
            actual = candidate.generation_provenance.checkpoint_id
            if actual != source_checkpoint_id:
                _fail(
                    "state_bank.candidate_checkpoint_provenance",
                    "candidate generation provenance belongs to another checkpoint",
                    event_id=record.event_id,
                    candidate_id=candidate.candidate_id,
                    expected=source_checkpoint_id,
                    actual=actual,
                )


def _validate_images(records: tuple[StateBankEvent, ...]) -> None:
    validated: set[tuple[Path, str, int, int]] = set()
    for record in records:
        image = record.image
        key = (image.path, image.content_sha256, image.width, image.height)
        if key in validated:
            continue
        if not image.path.is_file():
            _fail(
                "state_bank.image_missing",
                "state-bank image does not exist",
                path=str(image.path),
            )
        actual_sha = sha256_file(image.path)
        if actual_sha != image.content_sha256:
            _fail(
                "state_bank.image_hash",
                "state-bank image checksum differs from reviewed identity",
                path=str(image.path),
                expected=image.content_sha256,
                actual=actual_sha,
            )
        with Image.open(image.path) as opened:
            actual_size = tuple(int(value) for value in opened.size)
        if actual_size != (image.width, image.height):
            _fail(
                "state_bank.image_dimensions",
                "state-bank image dimensions differ from reviewed identity",
                path=str(image.path),
                expected=[image.width, image.height],
                actual=list(actual_size),
            )
        validated.add(key)


def _validation_receipt(
    manifest: StateBankManifest, records: tuple[StateBankEvent, ...]
) -> StateBankValidationReceipt:
    split_counts, family_counts = _record_counts(records)
    return StateBankValidationReceipt(
        bank_id=manifest.bank_id,
        source_checkpoint_id=manifest.source_checkpoint_id,
        source_checkpoint=manifest.source_checkpoint,
        records_sha256=manifest.records_sha256,
        record_count=len(records),
        split_counts=freeze_json(split_counts),
        event_family_counts=freeze_json(family_counts),
        rejection_reasons=manifest.rejection_reasons,
    )


def _record_counts(
    records: tuple[StateBankEvent, ...],
) -> tuple[dict[str, int], dict[str, int]]:
    split_counts = Counter(record.split for record in records)
    family_counts = Counter()
    for record in records:
        if record.entity_transition_eligible:
            family_counts["entity_transition"] += 1
        if record.coordinate_boundary_eligible:
            family_counts["coordinate_boundary"] += 1
        if (
            not record.entity_transition_eligible
            and not record.coordinate_boundary_eligible
        ):
            family_counts["diagnostic_only"] += 1
    return dict(sorted(split_counts.items())), dict(sorted(family_counts.items()))


def _split_assignments(
    records: tuple[StateBankEvent, ...],
) -> tuple[dict[str, Any], ...]:
    assignments: dict[int, dict[str, Any]] = {}
    for record in records:
        assignments.setdefault(
            record.image.image_id,
            {
                "image_id": record.image.image_id,
                "split": record.split,
                "split_group_id": record.split_group_id,
                "image_content_sha256": record.image.content_sha256,
            },
        )
    return tuple(assignments[image_id] for image_id in sorted(assignments))


def _split_assignment(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    checked = _mapping(value, field=field)
    _require_exact_keys(
        checked,
        {"image_id", "split", "split_group_id", "image_content_sha256"},
        field=field,
    )
    image_id = _image_id(checked["image_id"], field=f"{field}.image_id")
    _reject_blind_image_id(image_id, field=f"{field}.image_id")
    result = {
        "image_id": image_id,
        "split": _choice(checked["split"], _SPLITS, field=f"{field}.split"),
        "split_group_id": _string(
            checked["split_group_id"], field=f"{field}.split_group_id"
        ),
        "image_content_sha256": _string(
            checked["image_content_sha256"], field=f"{field}.image_content_sha256"
        ),
    }
    _require_sha256(
        result["image_content_sha256"], field=f"{field}.image_content_sha256"
    )
    if result["split_group_id"] != f"image:{image_id}":
        _fail(
            "state_bank.split_group",
            "manifest split group is not image-derived",
            field=field,
        )
    return result


def _source_artifact(value: Mapping[str, Any], *, field: str) -> dict[str, str]:
    checked = _mapping(value, field=field)
    _require_exact_keys(checked, {"artifact_id", "sha256"}, field=field)
    result = {
        "artifact_id": _string(checked["artifact_id"], field=f"{field}.artifact_id"),
        "sha256": _string(checked["sha256"], field=f"{field}.sha256"),
    }
    _require_sha256(result["sha256"], field=f"{field}.sha256")
    return result


def _reject_blind_sources(
    rollout_rows: Sequence[Mapping[str, Any]], review_rows: Sequence[Mapping[str, Any]]
) -> None:
    for owner, rows in (("rollout_rows", rollout_rows), ("review_rows", review_rows)):
        for index, row in enumerate(rows):
            image_value = row.get("image_id")
            image_mapping = row.get("image")
            if image_value is None and isinstance(image_mapping, Mapping):
                image_value = image_mapping.get("image_id")
            if image_value is not None:
                _reject_blind_image_id(
                    _image_id(image_value, field=f"{owner}[{index}].image_id"),
                    field=f"{owner}[{index}].image_id",
                )


def _reject_blind_image_id(image_id: int, *, field: str) -> None:
    if image_id in BLIND_IMAGE_IDS:
        _fail(
            "state_bank.blind_cohort",
            "blind evaluation image is forbidden from state-bank assembly and loading",
            field=field,
            image_id=image_id,
        )


def _checkpoint_identity(
    value: CheckpointIdentity | Mapping[str, Any],
) -> CheckpointIdentity:
    if isinstance(value, CheckpointIdentity):
        return value
    return CheckpointIdentity.from_mapping(_mapping(value, field="source_checkpoint"))


def _selected_sites(
    value: Any, *, field: str, token_count: int
) -> tuple[SelectedSite, ...]:
    declarations = [
        SelectedSite.from_mapping(item, field=f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field=field))
    ]
    by_offset: dict[int, SelectedSite] = {}
    for site in declarations:
        if site.candidate_token_offset >= token_count:
            _fail(
                "state_bank.selected_site_bounds",
                "selected site offset exceeds candidate token count",
                field=field,
                offset=site.candidate_token_offset,
                token_count=token_count,
            )
        prior = by_offset.get(site.candidate_token_offset)
        if prior is not None and prior.intended_token_type != site.intended_token_type:
            _fail(
                "state_bank.selected_site_conflict",
                "one candidate site declares conflicting intended token types",
                field=field,
                offset=site.candidate_token_offset,
                first=prior.intended_token_type,
                second=site.intended_token_type,
            )
        by_offset[site.candidate_token_offset] = site
    return tuple(by_offset[offset] for offset in sorted(by_offset))


def _coordinate_values(value: Any, *, field: str) -> tuple[int, ...]:
    values = tuple(
        _coordinate_value(item, field=f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field=field))
    )
    if not values:
        _fail(
            "state_bank.coordinate_acceptable_empty",
            "acceptable coordinate set must be nonempty",
            field=field,
        )
    if len(set(values)) != len(values):
        _fail(
            "state_bank.coordinate_acceptable_duplicate",
            "acceptable coordinate set must be unique",
            field=field,
        )
    return tuple(sorted(values))


def _coordinate_value(value: Any, *, field: str) -> int:
    parsed = _require_nonnegative_int(value, field=field)
    if parsed > 999:
        _fail(
            "state_bank.coordinate_range",
            "coordinate value must stay in [0, 999]",
            field=field,
            value=parsed,
        )
    return parsed


def _validate_token_hash(
    token_ids: tuple[int, ...], declared: str, *, field: str
) -> None:
    _require_sha256(declared, field=f"{field}_sha256")
    actual = token_ids_sha256(token_ids)
    if actual != declared:
        _fail(
            "state_bank.token_hash",
            "stored exact token identifiers do not match their declared hash",
            field=field,
            expected=declared,
            actual=actual,
        )


def _token_ids(value: Any, *, field: str, allow_empty: bool = False) -> tuple[int, ...]:
    result = tuple(
        _require_nonnegative_int(item, field=f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field=field))
    )
    if not result and not allow_empty:
        _fail(
            "state_bank.token_ids_empty",
            "exact token identifier sequence must be nonempty",
            field=field,
        )
    return result


def _interval(value: Any, *, field: str, upper_bound: int) -> tuple[int, int]:
    items = _sequence(value, field=field)
    if len(items) != 2:
        _fail(
            "state_bank.interval_shape",
            "token interval must contain start and end",
            field=field,
        )
    start = _require_nonnegative_int(items[0], field=f"{field}[0]")
    end = _require_nonnegative_int(items[1], field=f"{field}[1]")
    if start >= end or end > upper_bound:
        _fail(
            "state_bank.interval_bounds",
            "candidate-local half-open interval is empty or out of bounds",
            field=field,
            start=start,
            end=end,
            upper_bound=upper_bound,
        )
    return start, end


def _optional_interval(
    value: Any, *, field: str, upper_bound: int
) -> tuple[int, int] | None:
    return (
        None
        if value is None
        else _interval(value, field=field, upper_bound=upper_bound)
    )


def _image_id(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        _fail(
            "state_bank.image_id",
            "image id must be an integer or decimal string",
            field=field,
        )
    if isinstance(value, int):
        return _require_nonnegative_int(value, field=field)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    _fail(
        "state_bank.image_id",
        "image id must be an integer or decimal string",
        field=field,
        value=value,
    )


def _unique_rows_by_id(
    rows: Sequence[Mapping[str, Any]], *, field: str, owner: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, row_value in enumerate(rows):
        row = _mapping(row_value, field=f"{owner}[{index}]")
        identifier = _string(row.get(field), field=f"{owner}[{index}].{field}")
        if identifier in result:
            _fail(
                "state_bank.assembler_duplicate_id",
                "assembler identifiers must be unique",
                owner=owner,
                identifier=identifier,
            )
        result[identifier] = row
    return result


def _count_mapping(value: Any, *, field: str) -> dict[str, int]:
    mapping = _mapping(value, field=field)
    result: dict[str, int] = {}
    for key, count in mapping.items():
        reason = _string(key, field=f"{field}.key")
        result[reason] = _require_positive_int(count, field=f"{field}.{reason}")
    return dict(sorted(result.items()))


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = _strict_json_loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactContractError(
            "state-bank manifest does not exist",
            code="state_bank.manifest_missing",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ArtifactContractError(
            "state-bank manifest is not valid UTF-8 JSON",
            code="state_bank.manifest_json",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    return _mapping(payload, field="manifest")


def _strict_json_loads(payload: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_nonfinite_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number is forbidden: {value}")

    return json.loads(
        payload,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite_constant,
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(
            "state_bank.mapping",
            "field must be a JSON object",
            field=field,
            value_type=type(value).__name__,
        )
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(
            "state_bank.sequence",
            "field must be a JSON array",
            field=field,
            value_type=type(value).__name__,
        )
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, field: str
) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            "state_bank.fields",
            "state-bank object has missing or unknown fields",
            field=field,
            missing=sorted(expected - actual),
            unknown=sorted(actual - expected),
        )


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(
            "state_bank.string",
            "field must be a nonempty string",
            field=field,
            value_type=type(value).__name__,
        )
    return value.strip()


def _string_allow_empty(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        _fail(
            "state_bank.string",
            "field must be a string",
            field=field,
            value_type=type(value).__name__,
        )
    return value


def _optional_string(value: Any, *, field: str) -> str | None:
    return None if value is None else _string(value, field=field)


def _bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        _fail(
            "state_bank.bool",
            "field must be a boolean",
            field=field,
            value_type=type(value).__name__,
        )
    return value


def _choice(value: Any, allowed: frozenset[str], *, field: str) -> str:
    selected = _string(value, field=field)
    if selected not in allowed:
        _fail(
            "state_bank.choice",
            "field contains an unsupported value",
            field=field,
            value=selected,
            allowed=sorted(allowed),
        )
    return selected


def _optional_choice(value: Any, allowed: frozenset[str], *, field: str) -> str | None:
    return None if value is None else _choice(value, allowed, field=field)


def _require_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(
            "state_bank.nonnegative_int",
            "field must be a nonnegative integer",
            field=field,
            value=value,
        )
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    parsed = _require_nonnegative_int(value, field=field)
    if parsed <= 0:
        _fail(
            "state_bank.positive_int",
            "field must be a positive integer",
            field=field,
            value=value,
        )
    return parsed


def _finite_float(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(
            "state_bank.finite_float",
            "field must be a finite number",
            field=field,
            value_type=type(value).__name__,
        )
    parsed = float(value)
    if not math.isfinite(parsed):
        _fail(
            "state_bank.finite_float",
            "field must be a finite number",
            field=field,
            value=parsed,
        )
    if minimum is not None and parsed < minimum:
        _fail(
            "state_bank.finite_float_range",
            "field is below its minimum",
            field=field,
            value=parsed,
            minimum=minimum,
        )
    if maximum is not None and parsed > maximum:
        _fail(
            "state_bank.finite_float_range",
            "field exceeds its maximum",
            field=field,
            value=parsed,
            maximum=maximum,
        )
    return parsed


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(
            "state_bank.sha256",
            "field must be a lowercase SHA-256 digest",
            field=field,
            value=value,
        )
    return value


def _require_unique(values: Sequence[str], *, code: str, field: str) -> None:
    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        _fail(code, "identifiers must be unique", field=field, duplicates=duplicates)


def _fail(code: str, message: str, **context: Any) -> None:
    raise ArtifactContractError(message, code=code, context=context)


__all__ = [
    "BLIND_IMAGE_IDS",
    "CheckpointIdentity",
    "CoordinateBoundaryObservation",
    "CoordinateDecision",
    "GenerationProvenance",
    "ImageIdentity",
    "LoadedStateBank",
    "PhysicalEntity",
    "PrefixCoveredOwnerProof",
    "ReviewProvenance",
    "STATE_BANK_MANIFEST_NAME",
    "STATE_BANK_RECORDS_NAME",
    "STATE_BANK_SCHEMA_VERSION",
    "SelectedSite",
    "StateBankCandidate",
    "StateBankEvent",
    "StateBankManifest",
    "StateBankManifestBinding",
    "StateBankValidationReceipt",
    "assemble_state_bank",
    "load_state_bank",
    "load_state_bank_manifest_binding",
    "validate_state_bank_token_identity",
]
