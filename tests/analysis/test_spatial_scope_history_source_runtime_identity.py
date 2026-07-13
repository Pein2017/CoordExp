from __future__ import annotations

from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest

import src.analysis.spatial_scope_history.calibration as calibration
from src.analysis.spatial_scope_history.calibration import (
    FROZEN_CHECKPOINT_MANIFEST_SHA256,
    FROZEN_READINESS_LEDGER_SEAL_SHA256,
    FROZEN_RESOLVED_INFERENCE_CONFIG_SHA256,
    SourceRuntimeIdentityReceipt,
)
from src.analysis.spatial_scope_history.cohort_ledger import sha256_payload
from src.analysis.spatial_scope_history.spatial import SpatialGridSpec
from src.common.errors import ArtifactContractError

from scripts.research import materialize_source_runtime_identity as writer


def _committed_source_identity_without_worktree_cleanliness_gate():
    repository_root = calibration._source_runtime_repository_root()
    git_head_commit = (
        calibration._run_git(
            "rev-parse", "--verify", "HEAD", repository_root=repository_root
        )
        .decode("ascii")
        .strip()
    )
    hashes = []
    for relative_path in calibration._SOURCE_RUNTIME_CRITICAL_PATHS:
        committed_bytes = calibration._run_git(
            "show",
            f"{git_head_commit}:{relative_path}",
            repository_root=repository_root,
        )
        hashes.append((relative_path, hashlib.sha256(committed_bytes).hexdigest()))
    return git_head_commit, tuple(hashes)


@pytest.fixture(autouse=True)
def _isolate_from_parallel_critical_source_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    committed_identity = _committed_source_identity_without_worktree_cleanliness_gate()
    monkeypatch.setattr(
        calibration,
        "_live_committed_source_identity",
        lambda: committed_identity,
    )


def _digest(label: str) -> str:
    return sha256_payload({"label": label})


def _verified_shape_receipt() -> SourceRuntimeIdentityReceipt:
    source_attestation_sha256 = hashlib.sha256(
        calibration._live_source_attestation_bytes()
    ).hexdigest()
    git_head_commit, critical_source_sha256s = (
        calibration._live_committed_source_identity()
    )
    aggregate_sha256 = _digest("aggregate-artifact")
    aggregate_fingerprint = _digest("aggregate-payload")
    code_sha256 = calibration._derived_source_code_sha256(
        git_head_commit=git_head_commit,
        critical_source_sha256s=critical_source_sha256s,
        source_attestation_sha256=source_attestation_sha256,
    )
    runtime_sha256 = calibration._derived_source_runtime_sha256(
        source_attestation_sha256=source_attestation_sha256,
        aggregate_sha256=aggregate_sha256,
        aggregate_fingerprint=aggregate_fingerprint,
        config_sha256=FROZEN_RESOLVED_INFERENCE_CONFIG_SHA256,
        checkpoint_manifest_sha256=FROZEN_CHECKPOINT_MANIFEST_SHA256,
    )
    return SourceRuntimeIdentityReceipt(
        checkpoint_manifest_sha256=FROZEN_CHECKPOINT_MANIFEST_SHA256,
        code_sha256=code_sha256,
        config_sha256=FROZEN_RESOLVED_INFERENCE_CONFIG_SHA256,
        critical_source_sha256s=critical_source_sha256s,
        git_head_commit=git_head_commit,
        ledger_seal_sha256=FROZEN_READINESS_LEDGER_SEAL_SHA256,
        processor_contract_sha256=SpatialGridSpec().fingerprint,
        runtime_sha256=runtime_sha256,
        sampled_runtime_attestation_aggregate_fingerprint=aggregate_fingerprint,
        sampled_runtime_attestation_aggregate_sha256=aggregate_sha256,
        source_attestation_sha256=source_attestation_sha256,
    )


def test_source_runtime_identity_round_trip_revalidates_live_source() -> None:
    receipt = _verified_shape_receipt()

    loaded = SourceRuntimeIdentityReceipt.from_artifact_dict(receipt.to_artifact_dict())

    assert loaded == receipt


@pytest.mark.parametrize(
    ("field", "replacement", "error_match"),
    [
        ("code_sha256", _digest("forged-code"), "not verifier-derived"),
        ("runtime_sha256", _digest("forged-runtime"), "not verifier-derived"),
        ("config_sha256", _digest("forged-config"), "another resolved"),
        (
            "checkpoint_manifest_sha256",
            _digest("forged-checkpoint"),
            "another checkpoint",
        ),
    ],
)
def test_source_runtime_identity_rejects_forged_identity_fields(
    field: str,
    replacement: str,
    error_match: str,
) -> None:
    receipt = _verified_shape_receipt()

    with pytest.raises(ArtifactContractError, match=error_match):
        replace(receipt, **{field: replacement, "receipt_sha256": None})


def test_source_runtime_identity_rejects_unknown_fields() -> None:
    payload = _verified_shape_receipt().to_artifact_dict()
    payload["unverified_runtime_note"] = "accepted-by-self-reseal"

    with pytest.raises(ArtifactContractError, match="keys differ"):
        SourceRuntimeIdentityReceipt.from_artifact_dict(payload)


def test_source_runtime_identity_load_rejects_live_commit_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _verified_shape_receipt()
    monkeypatch.setattr(
        calibration,
        "_live_committed_source_identity",
        lambda: ("0" * 40, receipt.critical_source_sha256s),
    )

    with pytest.raises(ArtifactContractError, match="committed source"):
        SourceRuntimeIdentityReceipt.from_artifact_dict(receipt.to_artifact_dict())


def test_source_runtime_identity_writer_binds_inputs_and_refuses_overwrite(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _verified_shape_receipt()
    source_attestation = tmp_path / "source-attestation.json"
    aggregate = tmp_path / "sampled-runtime-attestation-aggregate.json"
    config = tmp_path / "resolved-inference-config.yaml"
    checkpoint = tmp_path / "checkpoint-manifest.json"
    ledger = tmp_path / "ledger-seal.json"
    output = tmp_path / "source-runtime-identity-receipt.json"
    for path, content in (
        (source_attestation, b"source"),
        (aggregate, b"aggregate"),
        (config, b"config"),
        (checkpoint, b"checkpoint"),
        (ledger, b"ledger"),
    ):
        path.write_bytes(content)
    captured: dict[str, object] = {}

    def derive(**kwargs):
        captured.update(kwargs)
        return receipt

    monkeypatch.setattr(
        writer.SourceRuntimeIdentityReceipt,
        "from_verified_artifacts",
        staticmethod(derive),
    )
    arguments = SimpleNamespace(
        source_attestation=source_attestation,
        sampled_runtime_attestation_aggregate=aggregate,
        resolved_inference_config=config,
        checkpoint_manifest=checkpoint,
        readiness_ledger_seal=ledger,
        output=output,
    )

    assert writer.run(arguments) == receipt
    assert captured == {
        "checkpoint_manifest_path": checkpoint.resolve(),
        "ledger_seal_sha256": hashlib.sha256(b"ledger").hexdigest(),
        "processor_contract_sha256": SpatialGridSpec().fingerprint,
        "resolved_inference_config_path": config.resolve(),
        "sampled_runtime_attestation_aggregate_path": aggregate.resolve(),
        "source_attestation_path": source_attestation.resolve(),
    }
    assert (
        SourceRuntimeIdentityReceipt.from_artifact_dict(
            calibration.load_canonical_json(output)
        )
        == receipt
    )

    with pytest.raises(ArtifactContractError, match="overwrite"):
        writer.run(arguments)
