"""Shadow config resolver for typed CoordExp training surfaces."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from src.training.pipelines.base import TrainingPipeline
from src.training.pipelines.stage1_compact_trie_ce import Stage1CompactTrieCEPipeline
from src.training.pipelines.stage1_json_ce import Stage1JsonCEPipeline
from src.training.pipelines.stage2_rollout_correction import (
    Stage2RolloutCorrectionPipeline,
)
from src.training.supervision.distributions import (
    TargetObjectiveId,
    validate_target_objective_id,
)


CANONICAL_TOP_LEVEL_DOMAINS: tuple[str, ...] = (
    "run",
    "surface",
    "data",
    "template",
    "supervision",
    "objectives",
    "observability",
    "artifacts",
    "runtime",
    "experimental",
)
REQUIRED_TOP_LEVEL_DOMAINS: frozenset[str] = frozenset(
    domain for domain in CANONICAL_TOP_LEVEL_DOMAINS if domain != "experimental"
)


def _removed_key(*parts: str) -> str:
    """Return a removed mechanism key from semantic fragments."""

    return "_".join(parts)


REMOVED_MECHANISM_KEYS: frozenset[str] = frozenset(
    (
        _removed_key("adjacent", "repulsion"),
        _removed_key("adjacent", "repulsion", "copy", "margin"),
        _removed_key("adjacent", "repulsion", "filter", "mode"),
        _removed_key("adjacent", "repulsion", "margin", "ratio"),
        _removed_key("adjacent", "repulsion", "weight"),
        _removed_key("duplicate", "burst", "unlikelihood"),
        _removed_key("eos", "loosen"),
        _removed_key("eos", "stop", "weight"),
        _removed_key("eos", "trust"),
        _removed_key("eos", "trust", "weight"),
        _removed_key("eos", "weighted", "loss"),
        _removed_key(
            "force",
            "continuation",
        ),
        _removed_key(
            "forced",
            "continuation",
        ),
        _removed_key("continue", "over", "eos", "margin"),
        _removed_key("continue", "over", "eos", "weight"),
        _removed_key("loss", "duplicate", "burst", "unlikelihood"),
        _removed_key("missing", "label", "prior", "weighted", "ce"),
        _removed_key("separator", "continue", "weight"),
        _removed_key("stop", "gate"),
        _removed_key("stop", "signal", "ce"),
        _removed_key("stop", "signal", "damping"),
    )
)
ORDERED_OBJECTIVE_IDS: tuple[str, ...] = (
    "token_ce",
    "trie_ce",
    "coord_soft_ce",
    "box_regression",
    "teacher_forcing",
)


@dataclass(frozen=True, slots=True)
class ExperimentalConfig:
    """Strict temporary escape hatch for shadow surface configs.

    :param owner: Accountable owner for the temporary option.
    :param expiry: Expiry date or milestone for removing the temporary option.
    :param notes: Human-readable rationale for the temporary option.
    :param surface_or_pipeline_opt_in: Explicit opt-in gate for the escape hatch.
    """

    owner: str
    expiry: str
    notes: str
    surface_or_pipeline_opt_in: bool

    def __post_init__(self) -> None:
        """Validate the authored experimental block."""

        # validate required string metadata.
        for field_name in ("owner", "expiry", "notes"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"experimental.{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())

        # require explicit opt-in rather than permissive custom extras.
        if type(self.surface_or_pipeline_opt_in) is not bool:
            raise TypeError("experimental.surface_or_pipeline_opt_in must be a boolean")
        if self.surface_or_pipeline_opt_in is not True:
            raise ValueError(
                "experimental.surface_or_pipeline_opt_in must be true for "
                "shadow surface or pipeline experiments"
            )


@dataclass(frozen=True, slots=True)
class ResolvedObjectiveEntry:
    """Resolved objective entry from keyed objective authoring.

    :param objective_id: Canonical objective identifier.
    :param enabled: Whether the objective contributes to runtime loss.
    :param weight: Objective-local contribution weight.
    :param config: Frozen objective-local scalar metadata.
    """

    objective_id: TargetObjectiveId
    enabled: bool
    weight: float
    config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze one resolved objective entry."""

        # normalize objective identity through the semantic objective registry.
        object.__setattr__(
            self,
            "objective_id",
            validate_target_objective_id(self.objective_id),
        )

        # validate opt-in and weight separately so disabled entries stay explicit.
        if type(self.enabled) is not bool:
            raise TypeError(f"objectives.{self.objective_id}.enabled must be a boolean")
        if not isinstance(self.weight, (int, float)) or isinstance(self.weight, bool):
            raise TypeError(f"objectives.{self.objective_id}.weight must be numeric")
        if not math.isfinite(float(self.weight)) or self.weight < 0:
            raise ValueError(
                f"objectives.{self.objective_id}.weight must be finite and >= 0"
            )
        object.__setattr__(self, "weight", float(self.weight))

        # freeze scalar objective metadata.
        object.__setattr__(
            self,
            "config",
            _freeze_scalar_mapping(
                self.config,
                path=f"objectives.{self.objective_id}.config",
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedObjectiveProfile:
    """Deterministically ordered objective profile."""

    objectives: tuple[ResolvedObjectiveEntry, ...]

    @property
    def enabled_objectives(self) -> tuple[ResolvedObjectiveEntry, ...]:
        """Objective entries with runtime loss enabled."""

        return tuple(entry for entry in self.objectives if entry.enabled)


@dataclass(frozen=True, slots=True)
class SurfaceObjectivePolicy:
    """Objective compatibility policy for one shadow training surface.

    :param allowed_objectives: Objectives that may be authored for the surface.
    :param required_enabled_objectives: Objectives that must be enabled.
    """

    allowed_objectives: frozenset[str]
    required_enabled_objectives: frozenset[str]

    def validate(
        self,
        *,
        surface_id: str,
        profile: ResolvedObjectiveProfile,
    ) -> None:
        """Validate that an objective profile matches one surface contract."""

        # reject objective/profile drift before downstream span realization.
        authored = {entry.objective_id for entry in profile.objectives}
        unsupported = sorted(authored - self.allowed_objectives)
        if unsupported:
            raise ValueError(
                f"surface {surface_id!r} does not support objective keys: "
                f"{unsupported}"
            )

        # require the semantic primitive that gives the surface its meaning.
        enabled = {entry.objective_id for entry in profile.enabled_objectives}
        missing = sorted(self.required_enabled_objectives - enabled)
        if missing:
            raise ValueError(
                f"surface {surface_id!r} requires enabled objectives: {missing}"
            )


SURFACE_OBJECTIVE_POLICIES: Mapping[str, SurfaceObjectivePolicy] = MappingProxyType(
    {
        "stage1_json_ce": SurfaceObjectivePolicy(
            allowed_objectives=frozenset(("token_ce",)),
            required_enabled_objectives=frozenset(("token_ce",)),
        ),
        "stage1_compact_trie_ce": SurfaceObjectivePolicy(
            allowed_objectives=frozenset(
                ("token_ce", "trie_ce", "coord_soft_ce", "box_regression")
            ),
            required_enabled_objectives=frozenset(("trie_ce",)),
        ),
        "stage2_rollout_correction": SurfaceObjectivePolicy(
            allowed_objectives=frozenset(("teacher_forcing",)),
            required_enabled_objectives=frozenset(("teacher_forcing",)),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedTrainingRun:
    """Resolved shadow training run metadata.

    :param run: Frozen run-domain metadata.
    :param surface: Frozen surface-domain metadata.
    :param pipeline: Selected shadow training pipeline descriptor.
    :param objectives: Deterministically ordered objective profile.
    :param domains: Frozen top-level domain mappings.
    :param experimental: Optional strict experimental escape hatch.
    :param top_level_domains: Authored top-level domain order after validation.
    """

    run: Mapping[str, object]
    surface: Mapping[str, object]
    pipeline: TrainingPipeline
    objectives: ResolvedObjectiveProfile
    domains: Mapping[str, Mapping[str, object]]
    experimental: ExperimentalConfig | None
    top_level_domains: tuple[str, ...]


class TrainingSurfaceResolver:
    """Resolve new shadow surface configs without affecting legacy loaders."""

    def __init__(self) -> None:
        """Initialize the closed shadow surface registry."""

        self._pipelines: Mapping[str, TrainingPipeline] = MappingProxyType(
            {
                "stage1_json_ce": Stage1JsonCEPipeline(),
                "stage1_compact_trie_ce": Stage1CompactTrieCEPipeline(),
                "stage2_rollout_correction": Stage2RolloutCorrectionPipeline(),
            }
        )

    def resolve(self, payload: Mapping[str, Any]) -> ResolvedTrainingRun:
        """Resolve a raw shadow config mapping into typed run metadata."""

        # validate root shape and top-level contract.
        if not isinstance(payload, Mapping):
            raise TypeError("training surface config must be a mapping")
        self._reject_removed_mechanisms(payload, path="<root>")
        top_level_domains = self._validate_top_level_domains(payload)

        # validate strict domain mappings before surface-specific checks.
        domains = {
            domain: self._require_mapping(payload[domain], path=domain)
            for domain in REQUIRED_TOP_LEVEL_DOMAINS
        }
        surface_id = self._resolve_surface_id(domains["surface"])
        pipeline = self._resolve_pipeline(surface_id)

        # validate selected surface sections and shared objective profile.
        self._validate_shared_domains(domains)
        self._validate_surface_specific_domains(surface_id, domains)
        objectives = self.resolve_objectives(domains["objectives"])
        self._validate_surface_objective_profile(surface_id, objectives)
        frozen_domains = self._freeze_domains(domains)

        # validate optional escape hatch only when authored.
        experimental = None
        if "experimental" in payload:
            experimental = self._resolve_experimental(
                self._require_mapping(payload["experimental"], path="experimental")
            )

        return ResolvedTrainingRun(
            run=frozen_domains["run"],
            surface=frozen_domains["surface"],
            pipeline=pipeline,
            objectives=objectives,
            domains=MappingProxyType(frozen_domains),
            experimental=experimental,
            top_level_domains=top_level_domains,
        )

    def resolve_objectives(
        self,
        payload: Mapping[str, Any],
    ) -> ResolvedObjectiveProfile:
        """Resolve keyed objectives into deterministic canonical order."""

        # validate objective section shape and removed mechanisms first.
        mapping = self._require_mapping(payload, path="objectives")
        self._reject_removed_mechanisms(mapping, path="objectives")

        # build ordered entries from keyed authoring without list replacement traps.
        entries: list[ResolvedObjectiveEntry] = []
        for objective_id in ORDERED_OBJECTIVE_IDS:
            if objective_id not in mapping:
                continue
            entry_payload = self._require_mapping(
                mapping[objective_id],
                path=f"objectives.{objective_id}",
            )
            self._validate_allowed_keys(
                entry_payload,
                path=f"objectives.{objective_id}",
                allowed={"enabled", "weight", "config"},
            )
            entries.append(
                ResolvedObjectiveEntry(
                    objective_id=validate_target_objective_id(objective_id),
                    enabled=self._objective_enabled(
                        entry_payload,
                        objective_id=objective_id,
                    ),
                    weight=self._objective_weight(
                        entry_payload,
                        objective_id=objective_id,
                    ),
                    config=entry_payload.get("config", {}),
                )
            )

        # reject unsupported keys after canonical ordering has handled known ids.
        unknown = sorted(str(key) for key in mapping if key not in ORDERED_OBJECTIVE_IDS)
        if unknown:
            raise ValueError(f"unknown objective keys: {unknown}")
        if not entries:
            raise ValueError("objectives must define at least one objective")
        if not any(entry.enabled for entry in entries):
            raise ValueError("objectives must enable at least one objective")

        return ResolvedObjectiveProfile(objectives=tuple(entries))

    def _validate_surface_objective_profile(
        self,
        surface_id: str,
        objectives: ResolvedObjectiveProfile,
    ) -> None:
        """Validate objective compatibility for the selected surface."""

        # keep semantic surface ids from drifting away from loss primitives.
        try:
            policy = SURFACE_OBJECTIVE_POLICIES[surface_id]
        except KeyError as exc:
            raise ValueError(f"missing objective policy for surface {surface_id!r}") from exc

        policy.validate(surface_id=surface_id, profile=objectives)

    def _validate_top_level_domains(self, payload: Mapping[str, Any]) -> tuple[str, ...]:
        """Return authored canonical top-level domains after strict validation."""

        # reject unexpected top-level domains.
        allowed = set(CANONICAL_TOP_LEVEL_DOMAINS)
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise ValueError(f"Unknown top-level config domains: {unknown}")

        # require all non-experimental domains in the new shadow schema.
        missing = sorted(domain for domain in REQUIRED_TOP_LEVEL_DOMAINS if domain not in payload)
        if missing:
            raise ValueError(f"Missing required top-level config domains: {missing}")

        return tuple(domain for domain in CANONICAL_TOP_LEVEL_DOMAINS if domain in payload)

    def _validate_shared_domains(
        self,
        domains: Mapping[str, Mapping[str, object]],
    ) -> None:
        """Validate strict shared domain keys."""

        # keep surface descriptors closed while leaving scalar metadata lightweight.
        self._validate_allowed_keys(
            domains["run"],
            path="run",
            allowed={"id", "scope", "name", "seed", "tags"},
        )
        self._validate_allowed_keys(
            domains["surface"],
            path="surface",
            allowed={"id", "version", "description"},
        )
        self._validate_allowed_keys(
            domains["data"],
            path="data",
            allowed={"train_jsonl", "validation_jsonl", "eval_jsonl", "dataset_id"},
        )
        self._validate_allowed_keys(
            domains["template"],
            path="template",
            allowed={"id", "variant", "object_format"},
        )
        self._validate_allowed_keys(
            domains["observability"],
            path="observability",
            allowed={"level", "event_sinks", "diagnostics"},
        )
        self._validate_allowed_keys(
            domains["artifacts"],
            path="artifacts",
            allowed={"output_root", "logging_root", "artifact_subdir", "manifest"},
        )
        self._validate_allowed_keys(
            domains["runtime"],
            path="runtime",
            allowed={"trainer", "trainer_variant", "precision", "packing", "cache"},
        )

    def _validate_surface_specific_domains(
        self,
        surface_id: str,
        domains: Mapping[str, Mapping[str, object]],
    ) -> None:
        """Validate sections that are meaningful only for one surface."""

        # dispatch to the surface-local supervision contract.
        supervision = domains["supervision"]
        if surface_id == "stage1_json_ce":
            self._validate_stage1_json_supervision(supervision)
            return
        if surface_id == "stage1_compact_trie_ce":
            self._validate_stage1_compact_supervision(supervision)
            return
        if surface_id == "stage2_rollout_correction":
            self._validate_stage2_rollout_correction_supervision(supervision)
            return

        raise ValueError(f"unsupported surface.id: {surface_id!r}")

    def _validate_stage1_json_supervision(
        self,
        supervision: Mapping[str, object],
    ) -> None:
        """Validate Stage-1 JSON CE supervision metadata."""

        # reject Stage-2-only assignment sections on the JSON baseline surface.
        self._validate_allowed_keys(
            supervision,
            path="supervision",
            allowed={"mode", "source", "format"},
        )
        self._require_mode(supervision, expected="json_ce")

    def _validate_stage1_compact_supervision(
        self,
        supervision: Mapping[str, object],
    ) -> None:
        """Validate Stage-1 compact-full trie CE supervision metadata."""

        # keep compact-specific knobs separate from Stage-2 channel assignment.
        self._validate_allowed_keys(
            supervision,
            path="supervision",
            allowed={"mode", "source", "compact_full", "coordinate_format"},
        )
        self._require_mode(supervision, expected="compact_trie")

    def _validate_stage2_rollout_correction_supervision(
        self,
        supervision: Mapping[str, object],
    ) -> None:
        """Validate Stage-2 rollout-correction supervision metadata."""

        self._validate_allowed_keys(
            supervision,
            path="supervision",
            allowed={"mode", "assignment", "duplicate_filter", "target_ir"},
        )
        self._require_mode(supervision, expected="rollout_correction")

    def _resolve_surface_id(self, surface: Mapping[str, object]) -> str:
        """Return the selected surface identifier."""

        # validate surface.id before pipeline lookup.
        surface_id = surface.get("id")
        if type(surface_id) is not str or not surface_id.strip():
            raise ValueError("surface.id must be a non-empty string")

        return surface_id.strip()

    def _resolve_pipeline(self, surface_id: str) -> TrainingPipeline:
        """Return the pipeline descriptor registered for the surface."""

        # select from the closed shadow surface registry.
        try:
            return self._pipelines[surface_id]
        except KeyError as exc:
            supported = sorted(self._pipelines)
            raise ValueError(
                f"unsupported surface.id {surface_id!r}; use one of: {supported}"
            ) from exc

    def _resolve_experimental(
        self,
        experimental: Mapping[str, object],
    ) -> ExperimentalConfig:
        """Return the strict experimental escape hatch."""

        # require a closed escape-hatch schema.
        self._validate_allowed_keys(
            experimental,
            path="experimental",
            allowed={"owner", "expiry", "notes", "surface_or_pipeline_opt_in"},
        )
        required = ("owner", "expiry", "notes", "surface_or_pipeline_opt_in")
        missing = [field_name for field_name in required if field_name not in experimental]
        if missing:
            raise ValueError(f"experimental missing required fields: {missing}")

        return ExperimentalConfig(
            owner=experimental["owner"],  # type: ignore[arg-type]
            expiry=experimental["expiry"],  # type: ignore[arg-type]
            notes=experimental["notes"],  # type: ignore[arg-type]
            surface_or_pipeline_opt_in=experimental["surface_or_pipeline_opt_in"],  # type: ignore[arg-type]
        )

    def _objective_enabled(
        self,
        payload: Mapping[str, object],
        *,
        objective_id: str,
    ) -> bool:
        """Return the objective enabled flag."""

        # default authored objectives to enabled unless explicitly disabled.
        enabled = payload.get("enabled", True)
        if type(enabled) is not bool:
            raise TypeError(f"objectives.{objective_id}.enabled must be a boolean")

        return enabled

    def _objective_weight(
        self,
        payload: Mapping[str, object],
        *,
        objective_id: str,
    ) -> float:
        """Return the objective weight."""

        # default objective-local weight to one.
        weight = payload.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise TypeError(f"objectives.{objective_id}.weight must be numeric")
        if not math.isfinite(float(weight)) or weight < 0:
            raise ValueError(f"objectives.{objective_id}.weight must be finite and >= 0")

        return float(weight)

    def _freeze_domains(
        self,
        domains: Mapping[str, Mapping[str, object]],
    ) -> dict[str, Mapping[str, object]]:
        """Return immutable scalar-only domain mappings."""

        # preserve resolver output as typed metadata, not mutable caller payload.
        return {
            domain_name: _freeze_scalar_mapping(domain, path=domain_name)
            for domain_name, domain in domains.items()
        }

    def _require_mapping(self, value: object, *, path: str) -> Mapping[str, object]:
        """Return a string-keyed mapping."""

        # validate structural mapping shape.
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a mapping")
        for key in value:
            if type(key) is not str:
                raise TypeError(f"{path} keys must be strings")

        return value

    def _validate_allowed_keys(
        self,
        payload: Mapping[str, object],
        *,
        path: str,
        allowed: set[str],
    ) -> None:
        """Reject unknown keys for a strict schema section."""

        # emit dotted paths to match the existing strict schema style.
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            dotted = [f"{path}.{key}" for key in unknown]
            raise ValueError(f"Unknown {path} keys: {dotted}")

    def _require_mode(self, supervision: Mapping[str, object], *, expected: str) -> None:
        """Require a surface-specific supervision mode."""

        # keep surface identity and supervision mode aligned.
        mode = supervision.get("mode")
        if mode != expected:
            raise ValueError(f"supervision.mode must be {expected!r}")

    def _reject_removed_mechanisms(self, value: object, *, path: str) -> None:
        """Reject removed mechanism identifiers anywhere in a new config."""

        # walk mappings by key so removed objectives fail even if disabled.
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                if key_text in REMOVED_MECHANISM_KEYS:
                    raise ValueError(
                        f"removed training mechanism {key_text!r} is not supported "
                        f"in new surface configs at {child_path}"
                    )
                self._reject_removed_mechanisms(child, path=child_path)
            return

        # walk sequences while preserving string scalar behavior.
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                self._reject_removed_mechanisms(child, path=f"{path}[{index}]")
            return

        # reject scalar string references to removed mechanisms.
        if isinstance(value, str) and value in REMOVED_MECHANISM_KEYS:
            raise ValueError(
                f"removed training mechanism {value!r} is not supported "
                f"in new surface configs at {path}"
            )


def _freeze_scalar_mapping(
    payload: Mapping[str, object],
    *,
    path: str,
) -> Mapping[str, object]:
    """Return an immutable scalar-metadata mapping."""

    # validate scalar metadata recursively.
    frozen: dict[str, object] = {}
    for key, value in payload.items():
        if type(key) is not str:
            raise TypeError(f"{path} keys must be strings")
        frozen[key] = _freeze_scalar_value(value, path=f"{path}.{key}")

    return MappingProxyType(frozen)


def _freeze_scalar_value(value: object, *, path: str) -> object:
    """Return an immutable scalar metadata value."""

    # preserve simple scalar metadata as-is.
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value

    # freeze nested scalar maps and sequences for config metadata.
    if isinstance(value, Mapping):
        return _freeze_scalar_mapping(value, path=path)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_scalar_value(item, path=f"{path}[]") for item in value)

    raise TypeError(f"{path} must contain only scalar metadata")
