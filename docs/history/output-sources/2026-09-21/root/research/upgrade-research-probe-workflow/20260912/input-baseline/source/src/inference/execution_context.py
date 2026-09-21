"""Optional caller-owned opaque inference execution-context sidecar.

This module carries a caller-supplied opaque context (and an optional
immutable execution-journal plan reference) through inference assembly,
worker transport, and merge. It never interprets the context and never
predicts or requires a terminal fingerprint.

Preparation (``prepare_execution_context_payload``) is pure and does no
filesystem work, so an invalid caller context can be rejected before a run
directory, resolved-config artifact, or terminal-failure artifact is ever
published. Publication (``publish_execution_context_artifact``) is the only
step that touches disk.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.artifacts.json_values import (
    json_sha256,
    load_canonical_json,
    publish_json_exclusive,
    validate_json_value,
)
from src.common.errors import ArtifactContractError, RuntimeContractError


EXECUTION_CONTEXT_ARTIFACT_NAME = "execution_context.json"
EXECUTION_CONTEXT_SCHEMA_VERSION = 1
JOURNAL_PLAN_REFERENCE_KEYS = (
    "journal_schema_version",
    "execution_id",
    "plan_fingerprint",
    "plan_file_sha256",
)
_PAYLOAD_KEYS = (
    "execution_context_schema_version",
    "context",
    "context_fingerprint",
    "journal_plan_reference",
)
_IDENTITY_KEYS = (
    "locator",
    "relative_path",
    "file_sha256",
    "value_fingerprint",
    "journal_plan_reference",
)
_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ExecutionContextArtifact:
    """A published or verified controller-owned context byte source."""

    path: Path
    locator: str
    file_sha256: str
    value_fingerprint: str
    journal_plan_reference: dict[str, Any] | None
    payload: dict[str, Any]

    def identity(self) -> dict[str, Any]:
        return {
            "locator": self.locator,
            "relative_path": EXECUTION_CONTEXT_ARTIFACT_NAME,
            "file_sha256": self.file_sha256,
            "value_fingerprint": self.value_fingerprint,
            "journal_plan_reference": self.journal_plan_reference,
        }


def require_sha256_digest(value: Any, *, field: str) -> str:
    """Validate that ``value`` is a well-formed lowercase hex SHA-256 digest."""

    if not isinstance(value, str) or not _SHA256_HEX_PATTERN.match(value):
        raise ArtifactContractError(
            "execution-context digest must be a lowercase hex SHA-256 string",
            code="inference.execution_context_digest_format_invalid",
            context={"field": field, "value": value},
        )
    return value


def validate_journal_plan_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable pre-execution journal plan reference shape."""

    if not isinstance(reference, Mapping):
        raise ArtifactContractError(
            "journal plan reference must be a JSON object",
            code="inference.execution_context_plan_reference_invalid",
            context={"observed_type": type(reference).__name__},
        )
    validate_json_value(dict(reference))
    observed_keys = set(reference.keys())
    expected_keys = set(JOURNAL_PLAN_REFERENCE_KEYS)
    if observed_keys != expected_keys:
        raise ArtifactContractError(
            "journal plan reference must bind exactly the immutable"
            " pre-execution fields",
            code="inference.execution_context_plan_reference_invalid",
            context={
                "expected_keys": sorted(expected_keys),
                "observed_keys": sorted(observed_keys),
            },
        )
    schema_version = reference["journal_schema_version"]
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ArtifactContractError(
            "journal plan reference schema version must be an integer",
            code="inference.execution_context_plan_reference_invalid",
            context={"field": "journal_schema_version"},
        )
    execution_id = reference["execution_id"]
    if not isinstance(execution_id, str) or not execution_id:
        raise ArtifactContractError(
            "journal plan reference execution_id must be a non-empty string",
            code="inference.execution_context_plan_reference_invalid",
            context={"field": "execution_id"},
        )
    require_sha256_digest(reference["plan_fingerprint"], field="plan_fingerprint")
    require_sha256_digest(reference["plan_file_sha256"], field="plan_file_sha256")
    return dict(reference)


def build_execution_context_payload(
    *,
    context: Mapping[str, Any],
    journal_plan_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and canonicalize one caller-owned opaque execution context."""

    if not isinstance(context, Mapping):
        raise ArtifactContractError(
            "execution context must be a JSON object",
            code="inference.execution_context_invalid",
            context={"observed_type": type(context).__name__},
        )
    opaque_context = dict(context)
    validate_json_value(opaque_context)
    payload: dict[str, Any] = {
        "execution_context_schema_version": EXECUTION_CONTEXT_SCHEMA_VERSION,
        "context": opaque_context,
        "context_fingerprint": json_sha256(opaque_context),
        "journal_plan_reference": (
            None
            if journal_plan_reference is None
            else validate_journal_plan_reference(journal_plan_reference)
        ),
    }
    validate_json_value(payload)
    return payload


def prepare_execution_context_payload(
    *,
    execution_context: Mapping[str, Any] | None,
    journal_plan_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate a caller-owned context with no filesystem side effect.

    Callers MUST invoke this before creating a run directory or writing any
    other artifact, so an invalid context fails before anything is published.
    """

    if execution_context is None:
        if journal_plan_reference is not None:
            raise ArtifactContractError(
                "a journal plan reference requires an execution context",
                code="inference.execution_context_reference_without_context",
            )
        return None
    return build_execution_context_payload(
        context=execution_context,
        journal_plan_reference=journal_plan_reference,
    )


def publish_execution_context_artifact(
    *,
    run_dir: Path,
    payload: Mapping[str, Any] | None,
) -> ExecutionContextArtifact | None:
    """Durably publish one already-validated controller-owned context payload."""

    if payload is None:
        return None
    validated_payload = dict(payload)
    path = run_dir / EXECUTION_CONTEXT_ARTIFACT_NAME
    publish_json_exclusive(path, validated_payload)
    resolved_path = path.resolve()
    return ExecutionContextArtifact(
        path=resolved_path,
        locator=str(resolved_path),
        file_sha256=_file_sha256(path),
        value_fingerprint=json_sha256(validated_payload),
        journal_plan_reference=validated_payload["journal_plan_reference"],
        payload=validated_payload,
    )


def recover_published_execution_context_artifact(
    *,
    run_dir: Path,
    payload: Mapping[str, Any] | None,
) -> ExecutionContextArtifact | None:
    """Recover identity only when a failed publication left exact final bytes.

    This does not convert an uncertain publication into success.  It lets the
    caller bind an honest terminal-failure family to the final sidecar that was
    already created before a containing-directory sync error was reported.
    """

    if payload is None:
        return None
    path = (run_dir / EXECUTION_CONTEXT_ARTIFACT_NAME).resolve()
    if not path.is_file():
        return None
    expected_payload = dict(payload)
    observed_payload = load_canonical_json(path)
    if observed_payload != expected_payload:
        return None
    return ExecutionContextArtifact(
        path=path,
        locator=str(path),
        file_sha256=_file_sha256(path),
        value_fingerprint=json_sha256(observed_payload),
        journal_plan_reference=observed_payload["journal_plan_reference"],
        payload=observed_payload,
    )


def materialize_execution_context_artifact(
    *,
    run_dir: Path,
    execution_context: Mapping[str, Any] | None,
    journal_plan_reference: Mapping[str, Any] | None = None,
) -> ExecutionContextArtifact | None:
    """Validate then publish in one step (convenience for non-split callers)."""

    payload = prepare_execution_context_payload(
        execution_context=execution_context,
        journal_plan_reference=journal_plan_reference,
    )
    return publish_execution_context_artifact(run_dir=run_dir, payload=payload)


def load_and_verify_execution_context_artifact(
    *,
    locator: str,
    expected_file_sha256: str,
    expected_value_fingerprint: str,
    expected_journal_plan_reference: Mapping[str, Any] | None = None,
) -> ExecutionContextArtifact:
    """Load a controller-declared context file and verify exact agreement.

    Called by a worker before backend session creation. Validates the exact
    on-disk locator, the payload schema, the inner context-fingerprint
    self-consistency, the transported file/value digests, and the journal
    plan reference explicitly (not only transitively through a digest match).
    Fails closed on any disagreement instead of trusting the transported path
    alone.
    """

    if not locator:
        raise RuntimeContractError(
            "worker execution-context locator must be a non-empty path",
            code="inference.execution_context_locator_invalid",
            context={"locator": locator},
        )
    path = Path(locator)
    if not path.is_absolute():
        raise RuntimeContractError(
            "worker execution-context locator must be an absolute path",
            code="inference.execution_context_locator_invalid",
            context={"locator": locator},
        )
    require_sha256_digest(expected_file_sha256, field="expected_file_sha256")
    require_sha256_digest(
        expected_value_fingerprint, field="expected_value_fingerprint"
    )

    payload = load_canonical_json(path)
    _require_payload_schema(payload, locator=locator)

    observed_file_sha256 = _file_sha256(path)
    observed_value_fingerprint = json_sha256(payload)
    if (
        observed_file_sha256 != expected_file_sha256
        or observed_value_fingerprint != expected_value_fingerprint
    ):
        raise RuntimeContractError(
            "worker execution-context bytes disagree with the"
            " controller-declared digest",
            code="inference.execution_context_digest_mismatch",
            context={
                "locator": locator,
                "expected_file_sha256": expected_file_sha256,
                "observed_file_sha256": observed_file_sha256,
                "expected_value_fingerprint": expected_value_fingerprint,
                "observed_value_fingerprint": observed_value_fingerprint,
            },
        )
    reference = payload.get("journal_plan_reference")
    if reference is not None:
        validate_journal_plan_reference(reference)
    normalized_expected_reference = (
        None
        if expected_journal_plan_reference is None
        else validate_journal_plan_reference(expected_journal_plan_reference)
    )
    if reference != normalized_expected_reference:
        raise RuntimeContractError(
            "worker execution-context journal plan reference disagrees with"
            " the controller-declared launch contract",
            code="inference.execution_context_plan_reference_mismatch",
            context={
                "locator": locator,
                "expected_journal_plan_reference": normalized_expected_reference,
                "observed_journal_plan_reference": reference,
            },
        )
    return ExecutionContextArtifact(
        path=path,
        locator=locator,
        file_sha256=observed_file_sha256,
        value_fingerprint=observed_value_fingerprint,
        journal_plan_reference=reference,
        payload=payload,
    )


def publish_rank_local_execution_context_copy(
    *,
    payload: Mapping[str, Any],
    destination_dir: Path,
    expected_file_sha256: str,
) -> Path:
    """Durably publish a verified rank-local copy of the controller's context.

    Uses the same crash-consistent exclusive publication as the controller's
    own artifact (temp file, fsync, atomic link, directory fsync) instead of
    an unchecked ``write_bytes`` side file, then self-verifies the resulting
    bytes against the already-verified source digest.
    """

    destination = destination_dir / EXECUTION_CONTEXT_ARTIFACT_NAME
    publish_json_exclusive(destination, dict(payload))
    observed_file_sha256 = _file_sha256(destination)
    if observed_file_sha256 != expected_file_sha256:
        raise ArtifactContractError(
            "rank-local execution-context copy disagrees with the verified"
            " source bytes",
            code="inference.execution_context_copy_digest_mismatch",
            context={
                "path": str(destination),
                "expected_file_sha256": expected_file_sha256,
                "observed_file_sha256": observed_file_sha256,
            },
        )
    return destination


def validate_execution_context_identity(
    identity: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Defensively validate an already-resolved execution-context identity."""

    if identity is None:
        return None
    if not isinstance(identity, Mapping):
        raise ArtifactContractError(
            "execution-context identity must be a JSON object",
            code="inference.execution_context_identity_invalid",
            context={"observed_type": type(identity).__name__},
        )
    validate_json_value(dict(identity))
    observed_keys = set(identity.keys())
    expected_keys = set(_IDENTITY_KEYS)
    if observed_keys != expected_keys:
        raise ArtifactContractError(
            "execution-context identity must bind exactly the locator and"
            " digest fields",
            code="inference.execution_context_identity_invalid",
            context={
                "expected_keys": sorted(expected_keys),
                "observed_keys": sorted(observed_keys),
            },
        )
    locator = identity["locator"]
    if not isinstance(locator, str) or not locator:
        raise ArtifactContractError(
            "execution-context identity locator must be a non-empty string",
            code="inference.execution_context_identity_invalid",
            context={"field": "locator"},
        )
    relative_path = identity["relative_path"]
    if relative_path != EXECUTION_CONTEXT_ARTIFACT_NAME:
        raise ArtifactContractError(
            "execution-context identity relative path must be the fixed sidecar name",
            code="inference.execution_context_identity_invalid",
            context={
                "field": "relative_path",
                "expected": EXECUTION_CONTEXT_ARTIFACT_NAME,
                "observed": relative_path,
            },
        )
    require_sha256_digest(identity["file_sha256"], field="file_sha256")
    require_sha256_digest(identity["value_fingerprint"], field="value_fingerprint")
    reference = identity["journal_plan_reference"]
    if reference is not None:
        validate_journal_plan_reference(reference)
    return dict(identity)


def verify_local_execution_context_artifact(
    *,
    output_dir: Path,
    identity: Mapping[str, Any] | None,
    require_local_locator: bool = False,
) -> None:
    """Require the output-local sidecar to agree with the declared identity.

    The identity locator names the controller-owned source.  Direct execution
    uses that same file, while rank-local execution carries a byte-identical
    copy under ``output_dir``.  Artifact publication must validate the local
    bytes in both cases instead of trusting transport metadata alone.
    """

    validated_identity = validate_execution_context_identity(identity)
    local_path = (output_dir / EXECUTION_CONTEXT_ARTIFACT_NAME).resolve()
    if validated_identity is None:
        if local_path.exists():
            raise ArtifactContractError(
                "output directory has an undeclared execution-context sidecar",
                code="inference.execution_context_local_unexpected",
                context={"path": str(local_path)},
            )
        return
    if require_local_locator and validated_identity["locator"] != str(local_path):
        raise ArtifactContractError(
            "execution-context locator must name the output-local sidecar",
            code="inference.execution_context_local_locator_mismatch",
            context={
                "expected_locator": str(local_path),
                "declared_locator": validated_identity["locator"],
            },
        )
    load_and_verify_execution_context_artifact(
        locator=str(local_path),
        expected_file_sha256=validated_identity["file_sha256"],
        expected_value_fingerprint=validated_identity["value_fingerprint"],
        expected_journal_plan_reference=validated_identity["journal_plan_reference"],
    )


def _require_payload_schema(payload: Any, *, locator: str) -> None:
    if not isinstance(payload, dict) or set(payload.keys()) != set(_PAYLOAD_KEYS):
        raise RuntimeContractError(
            "worker execution-context payload does not match the required schema",
            code="inference.execution_context_schema_invalid",
            context={
                "locator": locator,
                "expected_keys": sorted(_PAYLOAD_KEYS),
                "observed_keys": (
                    sorted(payload.keys()) if isinstance(payload, dict) else None
                ),
            },
        )
    schema_version = payload["execution_context_schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != EXECUTION_CONTEXT_SCHEMA_VERSION
    ):
        raise RuntimeContractError(
            "worker execution-context payload declares an unsupported schema version",
            code="inference.execution_context_schema_invalid",
            context={
                "locator": locator,
                "expected_schema_version": EXECUTION_CONTEXT_SCHEMA_VERSION,
                "observed_schema_version": payload["execution_context_schema_version"],
            },
        )
    context = payload["context"]
    if not isinstance(context, dict):
        raise RuntimeContractError(
            "worker execution-context payload context must be a JSON object",
            code="inference.execution_context_schema_invalid",
            context={"locator": locator},
        )
    inner_context_fingerprint = json_sha256(context)
    if inner_context_fingerprint != payload["context_fingerprint"]:
        raise RuntimeContractError(
            "worker execution-context payload context_fingerprint disagrees"
            " with its own context value",
            code="inference.execution_context_inner_fingerprint_mismatch",
            context={
                "locator": locator,
                "declared_context_fingerprint": payload["context_fingerprint"],
                "recomputed_context_fingerprint": inner_context_fingerprint,
            },
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
