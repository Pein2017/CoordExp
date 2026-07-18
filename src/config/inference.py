"""Strict CoordExp-swift inference configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from src.common.errors import ConfigContractError
from src.config.fingerprint import sha256_file, sha256_json
from src.config.models import ConfigSource, PathOrigin, RunDirectory, StrictConfigModel
from src.config.paths import get_nested, set_nested


INFER_CONFIG_LOADER_VERSION = "coordexp-swift-infer-config-v1"
QUALIFIED_VLLM_VERSIONS = ("0.14.1",)
INFER_PATH_FIELDS = (
    "run.artifact_root",
    "model.base_model",
    "data.input_jsonl",
    "adapter.path",
    "embedding_delta.path",
)


class InferRunConfig(StrictConfigModel):
    name: str
    artifact_root: str
    output_dir: str | None = None
    collision_policy: Literal["fail", "timestamp"] = "fail"


class InferProcessorConfig(StrictConfigModel):
    do_resize: Literal[False] = False


class InferModelConfig(StrictConfigModel):
    base_model: str
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    processor: InferProcessorConfig = Field(default_factory=InferProcessorConfig)


class InferAdapterConfig(StrictConfigModel):
    type: Literal["dora"]
    path: str
    name: str = "default"


class InferEmbeddingDeltaConfig(StrictConfigModel):
    path: str


class InferDataConfig(StrictConfigModel):
    input_jsonl: str


class InferTemplatePromptConfig(StrictConfigModel):
    system: str | None = None
    user: str


class InferTemplateConfig(StrictConfigModel):
    object_field_order: Literal["desc_first", "geometry_first"]
    object_ordering: Literal["source_order", "geo_sorted", "random"]
    assistant_format: Literal["object_box_closed"]
    prompt: InferTemplatePromptConfig


class InferHfBackendOptions(StrictConfigModel):
    attn_implementation: Literal["flash_attention_2", "sdpa", "eager"]
    patch_embed_linearization: Literal["enabled", "disabled"]


class InferVllmBackendOptions(StrictConfigModel):
    gpu_memory_utilization: float = Field(
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )


class InferHfBackendConfig(StrictConfigModel):
    type: Literal["hf"]
    hf: InferHfBackendOptions


class InferVllmBackendConfig(StrictConfigModel):
    type: Literal["vllm"]
    vllm: InferVllmBackendOptions


InferBackendConfig = Annotated[
    InferHfBackendConfig | InferVllmBackendConfig,
    Field(discriminator="type"),
]


class InferGenerationConfig(StrictConfigModel):
    batch_size: int = Field(gt=0)
    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0, allow_inf_nan=False)
    top_p: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    n: Literal[1] = 1
    repetition_penalty: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)


class InferScoringConfig(StrictConfigModel):
    enabled: bool


class InferArtifactsConfig(StrictConfigModel):
    write_token_trace: bool = True
    write_parse_diagnostics: bool = True
    include_raw_model_logprob: bool = False


class InferDebugConfig(StrictConfigModel):
    smoke: bool = False
    dry_run: bool = False


class InferConfig(StrictConfigModel):
    schema_version: Literal[1]
    run: InferRunConfig
    model: InferModelConfig
    data: InferDataConfig
    template: InferTemplateConfig
    backend: InferBackendConfig
    generation: InferGenerationConfig
    scoring: InferScoringConfig
    artifacts: InferArtifactsConfig
    debug: InferDebugConfig = Field(default_factory=InferDebugConfig)
    adapter: InferAdapterConfig | None = None
    embedding_delta: InferEmbeddingDeltaConfig | None = None

    @model_validator(mode="after")
    def _production_batch_size_must_exceed_one(self) -> "InferConfig":
        if self.generation.batch_size == 1 and not (
            self.debug.smoke or self.debug.dry_run
        ):
            raise ConfigContractError(
                "generation.batch_size: 1 is allowed only for explicit debug or smoke inference",
                code="config.production_batch_size_one",
                context={
                    "generation.batch_size": 1,
                    "debug.smoke": self.debug.smoke,
                    "debug.dry_run": self.debug.dry_run,
                },
            )
        return self

    @model_validator(mode="after")
    def _canonical_inference_must_be_deterministic(self) -> "InferConfig":
        if self.generation.temperature != 0.0 or self.generation.top_p != 1.0:
            raise ConfigContractError(
                "canonical inference requires temperature: 0.0 and top_p: 1.0",
                code="config.deterministic_inference",
                context={
                    "generation.temperature": self.generation.temperature,
                    "generation.top_p": self.generation.top_p,
                },
            )
        return self

    @model_validator(mode="after")
    def _canonical_inference_requires_scoring(self) -> "InferConfig":
        if not self.scoring.enabled:
            raise ConfigContractError(
                "canonical inference requires scoring.enabled: true",
                code="config.scoring_required",
                context={"scoring.enabled": False},
            )
        return self

    @model_validator(mode="after")
    def _canonical_inference_requires_evidence(self) -> "InferConfig":
        disabled = [
            field
            for field, enabled in (
                ("artifacts.write_token_trace", self.artifacts.write_token_trace),
                (
                    "artifacts.write_parse_diagnostics",
                    self.artifacts.write_parse_diagnostics,
                ),
            )
            if not enabled
        ]
        if disabled:
            raise ConfigContractError(
                "canonical inference requires token trace and parse diagnostics evidence",
                code="config.inference_evidence_required",
                context={"disabled_fields": disabled},
            )
        return self


@dataclass(frozen=True)
class ResolvedInferConfig:
    config: InferConfig
    config_dict: dict[str, Any]
    fingerprint: str
    schema_version: int
    loader_version: str
    entry_config_path: Path
    sources: tuple[ConfigSource, ...]
    path_origins: dict[str, PathOrigin]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "config": self.config_dict,
            "resolution": {
                "schema_version": self.schema_version,
                "loader_version": self.loader_version,
                "fingerprint": self.fingerprint,
                "entry_config_path": str(self.entry_config_path),
                "sources": [source.to_artifact_dict() for source in self.sources],
                "path_origins": {
                    field: origin.to_artifact_dict()
                    for field, origin in sorted(self.path_origins.items())
                },
            },
        }


def load_infer_config(path: str | Path) -> ResolvedInferConfig:
    entry_path = Path(path).expanduser().resolve()
    _reject_legacy_infer_path(entry_path)
    merged, origins, sources = _load_with_extends(entry_path, stack=())
    if merged.get("schema_version") != 1:
        raise ConfigContractError(
            "inference config must declare schema_version: 1",
            code="config.schema_version",
            context={"path": str(entry_path)},
        )
    _reject_required_placeholders(merged)
    resolved_payload, path_origins = _resolve_infer_path_fields(merged, origins)
    try:
        config = InferConfig.model_validate(resolved_payload)
    except ConfigContractError:
        raise
    except ValidationError as exc:
        _raise_validation_error(exc, entry_path)
    _validate_canonical_namespace(config, entry_path)
    _validate_leaf_authorship(config, entry_path, origins)
    config_dict = config.model_dump(mode="json")
    fingerprint = sha256_json(config_dict)
    return ResolvedInferConfig(
        config=config,
        config_dict=config_dict,
        fingerprint=fingerprint,
        schema_version=config.schema_version,
        loader_version=INFER_CONFIG_LOADER_VERSION,
        entry_config_path=entry_path,
        sources=sources,
        path_origins=path_origins,
    )


def validate_vllm_runtime_version(*, observed_version: str) -> None:
    """Fail closed before engine construction for an unqualified vLLM version."""

    if observed_version not in QUALIFIED_VLLM_VERSIONS:
        raise ConfigContractError(
            "installed vLLM version is not runtime-qualified",
            code="config.vllm_version_unqualified",
            context={
                "observed_version": observed_version,
                "qualified_versions": list(QUALIFIED_VLLM_VERSIONS),
            },
        )


def resolve_infer_run_directory(
    config: InferConfig,
    *,
    cwd: Path | None = None,
    timestamp: str | None = None,
) -> RunDirectory:
    root_base = Path(config.run.artifact_root)
    root = root_base if root_base.is_absolute() else (cwd or Path.cwd()) / root_base
    root = root.resolve()

    run_dir_name = config.run.output_dir or config.run.name
    run_dir_path = Path(run_dir_name)
    if run_dir_path.is_absolute() or ".." in run_dir_path.parts:
        raise ConfigContractError(
            "run.output_dir must stay under run.artifact_root",
            code="config.run_output_dir_escape",
            context={"output_dir": run_dir_name},
        )
    run_dir = (root / run_dir_path).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise ConfigContractError(
            "run output directory escaped artifact root",
            code="config.run_output_dir_escape",
            context={"artifact_root": str(root), "run_dir": str(run_dir)},
            cause=exc,
        ) from exc

    if run_dir.exists():
        if config.run.collision_policy == "fail":
            raise ConfigContractError(
                "run output directory already exists",
                code="config.run_dir_exists",
                context={"run_dir": str(run_dir), "collision_policy": "fail"},
            )
        suffix = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = run_dir.with_name(f"{run_dir.name}-{suffix}")
        if run_dir.exists():
            raise ConfigContractError(
                "timestamped run output directory already exists",
                code="config.timestamped_run_dir_exists",
                context={"run_dir": str(run_dir), "collision_policy": "timestamp"},
            )

    return RunDirectory(
        run_name=config.run.name,
        artifact_root=root,
        run_dir=run_dir.resolve(),
        collision_policy=config.run.collision_policy,
    )


def _reject_legacy_infer_path(path: Path) -> None:
    parts = path.parts
    for index in range(len(parts) - 1):
        if parts[index] == "configs" and parts[index + 1] == "infer":
            raise ConfigContractError(
                "legacy configs/infer files are reference-only for CoordExp-swift V1 inference",
                code="config.legacy_infer_path",
                context={"path": str(path)},
            )


def _validate_canonical_namespace(config: InferConfig, entry_path: Path) -> None:
    if config.debug.smoke or config.debug.dry_run:
        return
    canonical_root = Path.cwd().resolve() / "configs" / "coordexp_swift" / "infer"
    try:
        entry_path.relative_to(canonical_root)
    except ValueError as exc:
        raise ConfigContractError(
            "production inference configs must live under configs/coordexp_swift/infer",
            code="config.noncanonical_infer_path",
            context={
                "path": str(entry_path),
                "canonical_root": str(canonical_root),
            },
            cause=exc,
        ) from exc


def _load_with_extends(
    path: Path,
    *,
    stack: tuple[Path, ...],
) -> tuple[dict[str, Any], dict[str, Path], tuple[ConfigSource, ...]]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigContractError(
            "config extends cycle detected",
            code="config.extends_cycle",
            context={"cycle": cycle},
        )
    payload = _load_yaml_mapping(path)
    _reject_nested_extends(payload, path)
    parent_ref = payload.pop("extends", None)
    if parent_ref is None:
        merged: dict[str, Any] = {}
        origins: dict[str, Path] = {}
        sources: tuple[ConfigSource, ...] = ()
    else:
        if not isinstance(parent_ref, str):
            raise ConfigContractError(
                "extends must be a single parent path string",
                code="config.extends_shape",
                context={"path": str(path), "value_type": type(parent_ref).__name__},
            )
        parent_path = (path.parent / parent_ref).resolve()
        merged, origins, sources = _load_with_extends(
            parent_path,
            stack=(*stack, path),
        )
    current_origins = _leaf_origins(payload, path)
    merged_payload, merged_origins = _deep_merge(
        merged,
        payload,
        origins,
        current_origins,
    )
    return (
        merged_payload,
        merged_origins,
        (*sources, ConfigSource(path=path, sha256=sha256_file(path))),
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigContractError(
            "config file does not exist",
            code="config.missing_file",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ConfigContractError(
            "config file must contain a YAML mapping",
            code="config.mapping",
            context={"path": str(path), "value_type": type(payload).__name__},
        )
    return payload


def _reject_nested_extends(value: Any, path: Path, dotted: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{dotted}.{key}" if dotted else str(key)
            if key == "extends" and dotted:
                raise ConfigContractError(
                    "extends is allowed only at the YAML top level",
                    code="config.nested_extends",
                    context={"path": str(path), "field": child_path},
                )
            _reject_nested_extends(child, path, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nested_extends(child, path, f"{dotted}[{index}]")


def _leaf_origins(payload: dict[str, Any], path: Path) -> dict[str, Path]:
    origins: dict[str, Path] = {}

    def visit(value: Any, parts: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*parts, str(key)))
        else:
            origins[".".join(parts)] = path

    visit(payload, ())
    return origins


def _deep_merge(
    parent: dict[str, Any],
    child: dict[str, Any],
    parent_origins: dict[str, Path],
    child_origins: dict[str, Path],
    *,
    prefix: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    merged = _deep_copy_dict(parent)
    origins = dict(parent_origins)
    for key, child_value in child.items():
        field_path = f"{prefix}.{key}" if prefix else key
        parent_value = merged.get(key)
        if isinstance(parent_value, dict) and isinstance(child_value, dict):
            sub_payload, sub_origins = _deep_merge(
                parent_value,
                child_value,
                _sub_origins(origins, key),
                _sub_origins(child_origins, key),
                prefix=field_path,
            )
            merged[key] = sub_payload
            _remove_origin_prefix(origins, key)
            origins.update(
                {
                    f"{key}.{sub_key}": sub_path
                    for sub_key, sub_path in sub_origins.items()
                }
            )
            continue
        if child_value is None and key in merged:
            raise ConfigContractError(
                "null cannot delete an inherited config value",
                code="config.null_inherited_delete",
                context={
                    "field": field_path,
                    "declaring_config_path": str(child_origins.get(key, "")),
                },
            )
        merged[key] = _deep_copy_value(child_value)
        _remove_origin_prefix(origins, key)
        origins.update(
            {
                path: origin
                for path, origin in child_origins.items()
                if path == key or path.startswith(f"{key}.")
            }
        )
    return merged, origins


def _sub_origins(origins: dict[str, Path], key: str) -> dict[str, Path]:
    prefix = f"{key}."
    return {
        path.removeprefix(prefix): origin
        for path, origin in origins.items()
        if path.startswith(prefix)
    }


def _remove_origin_prefix(origins: dict[str, Path], key: str) -> None:
    prefix = f"{key}."
    for path in list(origins):
        if path == key or path.startswith(prefix):
            del origins[path]


def _reject_required_placeholders(value: Any, dotted: str = "") -> None:
    if value == "REQUIRED":
        raise ConfigContractError(
            "REQUIRED placeholder survived into runnable config",
            code="config.required_placeholder",
            context={"field": dotted or "<root>"},
        )
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{dotted}.{key}" if dotted else str(key)
            _reject_required_placeholders(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_required_placeholders(child, f"{dotted}[{index}]")


def _raise_validation_error(exc: ValidationError, path: Path) -> None:
    errors = exc.errors()
    selected_error = errors[0] if errors else {}
    location = ".".join(str(part) for part in selected_error.get("loc", ()))
    validation_message = str(selected_error.get("msg", str(exc)))
    raise ConfigContractError(
        "config schema validation failed",
        code="config.schema_validation",
        context={
            "path": str(path),
            "field": location,
            "message": validation_message,
            "error_count": len(errors),
        },
        cause=exc,
    ) from exc


def _resolve_infer_path_fields(
    payload: dict[str, Any],
    leaf_origins: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, PathOrigin]]:
    resolved = _deep_copy_dict(payload)
    path_origins: dict[str, PathOrigin] = {}
    for field in INFER_PATH_FIELDS:
        declared = get_nested(resolved, field)
        if declared is None:
            continue
        if not isinstance(declared, str):
            raise ConfigContractError(
                "path field must be a string",
                code="config.path_type",
                context={"field": field, "value_type": type(declared).__name__},
            )
        declaring_file = leaf_origins.get(field)
        if declaring_file is None:
            continue
        declared_path = Path(declared)
        resolved_path = (
            declared_path
            if declared_path.is_absolute()
            else declaring_file.parent / declared_path
        ).resolve()
        set_nested(resolved, field, str(resolved_path))
        path_origins[field] = PathOrigin(
            field=field,
            declared_path=declared,
            declaring_config_path=declaring_file,
            resolved_path=resolved_path,
        )
    return resolved, path_origins


def _validate_leaf_authorship(
    config: InferConfig,
    entry_path: Path,
    origins: dict[str, Path],
) -> None:
    if config.debug.smoke:
        return
    required_leaf_fields = (
        "data.input_jsonl",
        "generation.batch_size",
        "generation.max_new_tokens",
        "generation.temperature",
        "generation.top_p",
        "scoring.enabled",
    )
    inherited = [
        field
        for field in required_leaf_fields
        if origins.get(field) is not None and origins[field].resolve() != entry_path
    ]
    if inherited:
        raise ConfigContractError(
            "production inference leaf must explicitly declare required runtime fields",
            code="config.production_leaf_required",
            context={"fields": inherited, "entry_config_path": str(entry_path)},
        )


def _deep_copy_dict(payload: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            copied[key] = _deep_copy_dict(value)
        elif isinstance(value, list):
            copied[key] = [_deep_copy_value(item) for item in value]
        else:
            copied[key] = value
    return copied


def _deep_copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _deep_copy_dict(value)
    if isinstance(value, list):
        return [_deep_copy_value(item) for item in value]
    return value
