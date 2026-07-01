"""Discoverable registry for typed CoordExp training pipelines."""

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
PIPELINE_IDS: tuple[str, ...] = (
    "stage1_standard_sft",
    "stage1_research_teacher_forcing",
    "stage2_rollout_correction",
)
CANONICAL_TOP_LEVEL_DOMAINS: tuple[str, ...] = (
    "run",
    "pipeline",
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
    "standard_ce",
    "research_teacher_forcing",
    "residual_set_correction",
)
REJECTED_PUBLIC_OBJECTIVE_IDS: Mapping[str, str] = MappingProxyType(
    {
        "teacher_forcing": (
            "objective key 'teacher_forcing' is not public registry behavior; "
            "use 'research_teacher_forcing' for active Stage-1 configs or "
            "'residual_set_correction' for Stage-2 rollout correction"
        ),
        "token_ce": (
            "objective key 'token_ce' is internal implementation/metric "
            "vocabulary; use public objective key 'standard_ce'"
        ),
    }
)
VALID_OBJECTIVE_IDS: frozenset[str] = frozenset(ORDERED_OBJECTIVE_IDS)


def _validate_objective_id(objective_id: object) -> str:
    """Return a registry objective id, rejecting retired public names."""

    if type(objective_id) is not str:
        raise TypeError("objective id must be a semantic string")
    if objective_id in REJECTED_PUBLIC_OBJECTIVE_IDS:
        raise ValueError(REJECTED_PUBLIC_OBJECTIVE_IDS[objective_id])
    if objective_id not in VALID_OBJECTIVE_IDS:
        supported = ", ".join(sorted(VALID_OBJECTIVE_IDS))
        raise ValueError(f"unknown objective key {objective_id!r}; use one of: {supported}")

    return objective_id


@dataclass(frozen=True, slots=True)
class ExperimentalConfig:
    """Strict temporary escape hatch for pipeline registry configs.

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
                "pipeline registry experiments"
            )


@dataclass(frozen=True, slots=True)
class ResolvedObjectiveEntry:
    """Resolved objective entry from keyed objective authoring.

    :param objective_id: Canonical objective identifier.
    :param enabled: Whether the objective contributes to runtime loss.
    :param weight: Objective-local contribution weight.
    :param config: Frozen objective-local scalar metadata.
    """

    objective_id: str
    enabled: bool
    weight: float
    config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze one resolved objective entry."""

        object.__setattr__(self, "objective_id", _validate_objective_id(self.objective_id))

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
class PipelineObjectivePolicy:
    """Objective compatibility policy for one training pipeline.

    :param allowed_objectives: Objectives that may be authored for the pipeline.
    :param required_enabled_objectives: Objectives that must be enabled.
    """

    allowed_objectives: frozenset[str]
    required_enabled_objectives: frozenset[str]

    def validate(
        self,
        *,
        pipeline_id: str,
        profile: ResolvedObjectiveProfile,
    ) -> None:
        """Validate that an objective profile matches one pipeline contract."""

        # reject objective/profile drift before downstream span realization.
        authored = {entry.objective_id for entry in profile.objectives}
        unsupported = sorted(authored - self.allowed_objectives)
        if unsupported:
            raise ValueError(
                f"pipeline {pipeline_id!r} does not support objective keys: "
                f"{unsupported}"
            )

        # require the semantic primitive that gives the pipeline its meaning.
        enabled = {entry.objective_id for entry in profile.enabled_objectives}
        missing = sorted(self.required_enabled_objectives - enabled)
        if missing:
            raise ValueError(
                f"pipeline {pipeline_id!r} requires enabled objectives: {missing}"
            )


PIPELINE_OBJECTIVE_POLICIES: Mapping[str, PipelineObjectivePolicy] = MappingProxyType(
    {
        "stage1_standard_sft": PipelineObjectivePolicy(
            allowed_objectives=frozenset(("standard_ce",)),
            required_enabled_objectives=frozenset(("standard_ce",)),
        ),
        "stage1_research_teacher_forcing": PipelineObjectivePolicy(
            allowed_objectives=frozenset(("research_teacher_forcing",)),
            required_enabled_objectives=frozenset(("research_teacher_forcing",)),
        ),
        "stage2_rollout_correction": PipelineObjectivePolicy(
            allowed_objectives=frozenset(("residual_set_correction",)),
            required_enabled_objectives=frozenset(("residual_set_correction",)),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedTrainingRun:
    """Resolved training pipeline metadata.

    :param run: Frozen run-domain metadata.
    :param pipeline_config: Frozen pipeline-domain metadata.
    :param pipeline: Selected training pipeline descriptor.
    :param objectives: Deterministically ordered objective profile.
    :param domains: Frozen top-level domain mappings.
    :param experimental: Optional strict experimental escape hatch.
    :param top_level_domains: Authored top-level domain order after validation.
    """

    run: Mapping[str, object]
    pipeline_config: Mapping[str, object]
    pipeline: TrainingPipeline
    objectives: ResolvedObjectiveProfile
    domains: Mapping[str, Mapping[str, object]]
    experimental: ExperimentalConfig | None
    top_level_domains: tuple[str, ...]


class TrainingPipelineRegistry:
    """Resolve and validate public CoordExp training pipeline identities."""

    def __init__(self) -> None:
        """Initialize the closed pipeline registry."""

        self._pipelines: Mapping[str, TrainingPipeline] = MappingProxyType(
            {
                "stage1_standard_sft": Stage1JsonCEPipeline(),
                "stage1_research_teacher_forcing": Stage1CompactTrieCEPipeline(),
                "stage2_rollout_correction": Stage2RolloutCorrectionPipeline(),
            }
        )

    def resolve(self, payload: Mapping[str, Any]) -> ResolvedTrainingRun:
        """Resolve a raw pipeline config mapping into typed run metadata."""

        # validate root shape and top-level contract.
        if not isinstance(payload, Mapping):
            raise TypeError("training pipeline config must be a mapping")
        if "surface" in payload:
            raise ValueError("surface.id is not public config; use top-level pipeline.id")
        self._reject_removed_mechanisms(payload, path="<root>")
        if self._is_target_hierarchy_payload(payload):
            payload = self._registry_payload_from_target_hierarchy(payload)
        top_level_domains = self._validate_top_level_domains(payload)

        # validate strict domain mappings before pipeline-specific checks.
        domains = {
            domain: self._require_mapping(payload[domain], path=domain)
            for domain in REQUIRED_TOP_LEVEL_DOMAINS
        }
        pipeline_id = self._resolve_pipeline_id(domains["pipeline"])
        pipeline = self._resolve_pipeline(pipeline_id)

        # validate selected pipeline sections and shared objective profile.
        self._validate_shared_domains(domains)
        self._validate_pipeline_specific_domains(pipeline_id, domains)
        objectives = self.resolve_objectives(domains["objectives"])
        self._validate_pipeline_objective_profile(pipeline_id, objectives)
        frozen_domains = self._freeze_domains(domains)

        # validate optional escape hatch only when authored.
        experimental = None
        if "experimental" in payload:
            experimental = self._resolve_experimental(
                self._require_mapping(payload["experimental"], path="experimental")
            )

        return ResolvedTrainingRun(
            run=frozen_domains["run"],
            pipeline_config=frozen_domains["pipeline"],
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
                    objective_id=_validate_objective_id(objective_id),
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

        # reject retired public names before generic unknown-key handling.
        rejected = sorted(str(key) for key in mapping if key in REJECTED_PUBLIC_OBJECTIVE_IDS)
        if rejected:
            raise ValueError("; ".join(REJECTED_PUBLIC_OBJECTIVE_IDS[key] for key in rejected))

        # reject unsupported keys after canonical ordering has handled known ids.
        unknown = sorted(str(key) for key in mapping if key not in ORDERED_OBJECTIVE_IDS)
        if unknown:
            raise ValueError(f"unknown objective keys: {unknown}")
        if not entries:
            raise ValueError("objectives must define at least one objective")
        if not any(entry.enabled for entry in entries):
            raise ValueError("objectives must enable at least one objective")

        return ResolvedObjectiveProfile(objectives=tuple(entries))

    def _validate_pipeline_objective_profile(
        self,
        pipeline_id: str,
        objectives: ResolvedObjectiveProfile,
    ) -> None:
        """Validate objective compatibility for the selected pipeline."""

        # keep semantic pipeline ids from drifting away from loss primitives.
        try:
            policy = PIPELINE_OBJECTIVE_POLICIES[pipeline_id]
        except KeyError as exc:
            raise ValueError(f"missing objective policy for pipeline {pipeline_id!r}") from exc

        policy.validate(pipeline_id=pipeline_id, profile=objectives)

    def _validate_top_level_domains(self, payload: Mapping[str, Any]) -> tuple[str, ...]:
        """Return authored canonical top-level domains after strict validation."""

        # reject the old public surface authoring path with migration guidance.
        if "surface" in payload:
            raise ValueError("surface.id is not public config; use top-level pipeline.id")

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

        # keep pipeline descriptors closed while leaving scalar metadata lightweight.
        self._validate_allowed_keys(
            domains["run"],
            path="run",
            allowed={"id", "scope", "name", "seed", "tags"},
        )
        self._validate_allowed_keys(
            domains["pipeline"],
            path="pipeline",
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
            allowed={"trainer", "precision", "packing", "cache"},
        )

    def _is_target_hierarchy_payload(self, payload: Mapping[str, Any]) -> bool:
        pipeline_raw = payload.get("pipeline")
        return isinstance(pipeline_raw, Mapping) and (
            "sample_factory" in payload
            or "detection_template" in payload
            or "token_embeddings_adapter" in payload
            or "stage2_rollout_correction" in payload
        )

    def _registry_payload_from_target_hierarchy(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, object]:
        pipeline_raw = self._require_mapping(payload.get("pipeline"), path="pipeline")
        pipeline_id = self._resolve_pipeline_id(pipeline_raw)
        data_raw = self._require_mapping(payload.get("data"), path="data")
        template_raw = self._require_mapping(
            payload.get("detection_template"),
            path="detection_template",
        )
        prompt_raw = payload.get("prompt", {})
        prompt = prompt_raw if isinstance(prompt_raw, Mapping) else {}
        training_raw = payload.get("training", {})
        training = training_raw if isinstance(training_raw, Mapping) else {}
        experiment_raw = payload.get("experiment", {})
        experiment = experiment_raw if isinstance(experiment_raw, Mapping) else {}

        if pipeline_id == "stage2_rollout_correction":
            objectives = self._stage2_objectives_from_target_hierarchy(payload)
            correction_raw = self._require_mapping(
                self._require_mapping(
                    payload.get("stage2_rollout_correction"),
                    path="stage2_rollout_correction",
                ).get("correction", {}),
                path="stage2_rollout_correction.correction",
            )
            assignment_raw = correction_raw.get("assignment", {})
            assignment = assignment_raw if isinstance(assignment_raw, Mapping) else {}
            assignment_strategy = str(assignment.get("strategy", "greedy_iou") or "")
            supervision = {
                "mode": "rollout_correction",
                "assignment": {"strategy": assignment_strategy},
                "duplicate_filter": {
                    "strategy": "rollout_correction_duplicate_control"
                },
                "target_ir": {"required": True},
            }
        else:
            objective_raw = self._require_mapping(
                payload.get("objective"),
                path="objective",
            )
            objective_id = _validate_objective_id(objective_raw.get("id"))
            objectives = {objective_id: {"enabled": True, "weight": 1.0}}
            supervision = {
                "mode": "compact_trie"
                if objective_id == "research_teacher_forcing"
                else "json_ce"
            }

        return {
            "run": {
                "id": str(training.get("run_name", pipeline_id) or pipeline_id),
                "scope": str(experiment.get("claim_scope", "active") or "active"),
            },
            "pipeline": {"id": pipeline_id},
            "data": {
                "train_jsonl": data_raw.get("train_jsonl"),
                "validation_jsonl": data_raw.get("val_jsonl"),
            },
            "template": {
                "id": template_raw.get("id"),
                "variant": prompt.get("variant"),
            },
            "supervision": supervision,
            "objectives": objectives,
            "observability": {"level": "minimal"},
            "artifacts": {
                "output_root": training.get("output_root", training.get("output_dir")),
                "logging_root": training.get("logging_root", training.get("logging_dir")),
                "artifact_subdir": training.get("artifact_subdir"),
            },
            "runtime": {"trainer": "pipeline"},
        }

    def _stage2_objectives_from_target_hierarchy(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, dict[str, object]]:
        stage2_raw = self._require_mapping(
            payload.get("stage2_rollout_correction"),
            path="stage2_rollout_correction",
        )
        pipeline_raw = self._require_mapping(
            stage2_raw.get("pipeline"),
            path="stage2_rollout_correction.pipeline",
        )
        objective_raw = pipeline_raw.get("objective")
        if not isinstance(objective_raw, Sequence) or isinstance(
            objective_raw, (str, bytes)
        ):
            raise TypeError("stage2_rollout_correction.pipeline.objective must be a list")
        objectives: dict[str, dict[str, object]] = {}
        for idx, item in enumerate(objective_raw):
            spec = self._require_mapping(
                item,
                path=f"stage2_rollout_correction.pipeline.objective[{idx}]",
            )
            objective_id = _validate_objective_id(spec.get("name"))
            objectives[objective_id] = {
                "enabled": bool(spec.get("enabled", True)),
                "weight": float(spec.get("weight", 1.0)),
                "config": spec.get("config", {}),
            }
        return objectives

    def _validate_pipeline_specific_domains(
        self,
        pipeline_id: str,
        domains: Mapping[str, Mapping[str, object]],
    ) -> None:
        """Validate sections that are meaningful only for one pipeline."""

        # dispatch to the pipeline-local supervision contract.
        supervision = domains["supervision"]
        if pipeline_id == "stage1_standard_sft":
            self._validate_stage1_json_supervision(supervision)
            return
        if pipeline_id == "stage1_research_teacher_forcing":
            self._validate_stage1_compact_supervision(supervision)
            return
        if pipeline_id == "stage2_rollout_correction":
            self._validate_stage2_rollout_correction_supervision(supervision)
            return

        raise ValueError(f"unsupported pipeline.id: {pipeline_id!r}")

    def _validate_stage1_json_supervision(
        self,
        supervision: Mapping[str, object],
    ) -> None:
        """Validate Stage-1 JSON CE supervision metadata."""

        # reject Stage-2-only assignment sections on the JSON baseline pipeline.
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
        assignment = self._require_mapping(
            supervision.get("assignment"),
            path="supervision.assignment",
        )
        self._validate_allowed_keys(
            assignment,
            path="supervision.assignment",
            allowed={"strategy"},
        )
        if assignment.get("strategy") != "greedy_iou":
            raise ValueError("supervision.assignment.strategy must be 'greedy_iou'")

        duplicate_filter = self._require_mapping(
            supervision.get("duplicate_filter"),
            path="supervision.duplicate_filter",
        )
        self._validate_allowed_keys(
            duplicate_filter,
            path="supervision.duplicate_filter",
            allowed={"strategy"},
        )
        if duplicate_filter.get("strategy") not in {
            "rollout_correction_duplicate_control",
            "deterministic_duplicate_filter",
        }:
            raise ValueError(
                "supervision.duplicate_filter.strategy must be "
                "'rollout_correction_duplicate_control' or "
                "'deterministic_duplicate_filter'"
            )

        target_ir = self._require_mapping(
            supervision.get("target_ir"),
            path="supervision.target_ir",
        )
        self._validate_allowed_keys(
            target_ir,
            path="supervision.target_ir",
            allowed={"required"},
        )
        if target_ir.get("required") is not True:
            raise ValueError("supervision.target_ir.required must be true")

    def _resolve_pipeline_id(self, pipeline: Mapping[str, object]) -> str:
        """Return the selected pipeline identifier."""

        # validate pipeline.id before descriptor lookup.
        pipeline_id = pipeline.get("id")
        if type(pipeline_id) is not str or not pipeline_id.strip():
            raise ValueError("pipeline.id must be a non-empty string")

        return pipeline_id.strip()

    def _resolve_pipeline(self, pipeline_id: str) -> TrainingPipeline:
        """Return the pipeline descriptor registered for the pipeline id."""

        # select from the closed pipeline registry.
        try:
            return self._pipelines[pipeline_id]
        except KeyError as exc:
            supported = sorted(self._pipelines)
            raise ValueError(
                f"unsupported pipeline.id {pipeline_id!r}; use one of: {supported}"
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
        """Require a pipeline-specific supervision mode."""

        # keep pipeline identity and supervision mode aligned.
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
                        f"in new pipeline configs at {child_path}"
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
                f"in new pipeline configs at {path}"
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
