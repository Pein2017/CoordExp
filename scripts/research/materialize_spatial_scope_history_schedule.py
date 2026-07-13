#!/usr/bin/env python3
"""Materialize one sealed Dense-Union-51 or Validation-200 primary schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.spatial_scope_history.calibration import (  # noqa: E402
    CalibrationSelectionReceipt,
    SourceRuntimeIdentityReceipt,
    load_attested_sampling_policy_set,
    load_canonical_json,
    materialize_primary_schedule_artifact,
    write_immutable_json,
)
from src.analysis.spatial_scope_history.cohort_ledger import (  # noqa: E402
    CohortLedger,
    sha256_file,
)
from src.analysis.spatial_scope_history.schedule import (  # noqa: E402
    PRIMARY_ROOT_SEED,
    SECOND_ROOT_SEED,
)
from src.analysis.spatial_scope_history.spatial import SpatialGridSpec  # noqa: E402


UNIT_ID = "2026-07-13-spatial-scope-history-disentanglement"
# Immutable readiness-v2 identity for this research unit.  The content
# declarations are pinned independently from the seal file hash so a
# self-consistently re-sealed ledger cannot become an alternate source.
FROZEN_READINESS_LEDGER_SEAL_SHA256 = (
    "c177bc2b12f06fed559660bb5e1d7ff24fa19ee5eb898e642cae27d6991ebb68"
)
FROZEN_READINESS_ARTIFACT_DIGESTS = {
    "adjudication.jsonl": (
        "dd84a2279b8a75955e3420440fd15c5e6811421e3ab3d5c3dc32b7cd37860421"
    ),
    "audit-augmented-ledger.jsonl": (
        "52e9f21eb32f7c3793d1356125931cb4c2a1fa5a12647d0d437dec217668d8df"
    ),
}
FROZEN_READINESS_CATEGORY_NAMESPACE_SHA256 = (
    "b7b3c8f2361189c88e103369fa7a6c207c81b79999f9610897cf949d546ea4fe"
)
FROZEN_READINESS_SOURCE_DIGESTS = {
    "adjudication_queue_jsonl": (
        "40ef7e15be040740e0e1fe92b5ecda2e5900e9353be5610d77301a846b365c3c"
    ),
    "official_crowd_ledger_jsonl": (
        "f5f3465f5fe44c873ff418176e8a5359f1e9a506c1adc6ff6296d0709e742ad5"
    ),
    "official_individual_ledger_jsonl": (
        "dfb1524ba8deb3c5c1fb5c1f5baf5f30f077ed6210d849a972fcca9817b75166"
    ),
    "pre_seal_ordering_correction_receipt_json": (
        "92df5006b5b7db586055a80914a2c80a34d67155732dff7e2245b14edf4fa9af"
    ),
    "review_queue_jsonl": (
        "3d24687456b775aef17fdce90ed6d2d553cf772421d59967c3aa87803cb70d59"
    ),
    "reviewer_one_labels_jsonl": (
        "d9149f023308e1e407ed54a6ff6693ba889e413abc753234044cb1dd38d90a52"
    ),
    "reviewer_two_labels_jsonl": (
        "6cf06dd001db509bb00f3cd657a010fec334e52c16f9d1b89d8e2c031b05f3b2"
    ),
}
_COHORT_FILES = {
    "dense-union-51": "dense-union-51-manifest.jsonl",
    "validation-200": "cohort-manifest.jsonl",
}
_SEALED_SOURCE_FILE_BY_KEY = {
    "adjudication_queue_jsonl": "adjudication-queue.jsonl",
    "official_crowd_ledger_jsonl": "official-crowd-ignore-ledger.jsonl",
    "official_individual_ledger_jsonl": "official-individual-ledger.jsonl",
    "pre_seal_ordering_correction_receipt_json": (
        "pre-seal-ordering-correction-receipt.json"
    ),
    "review_queue_jsonl": "review-queue.jsonl",
    "reviewer_one_labels_jsonl": "reviewer-one-labels.jsonl",
    "reviewer_two_labels_jsonl": "reviewer-two-labels.jsonl",
}


def _validate_readiness_root(readiness_root: Path) -> tuple[Path, dict[str, str]]:
    root = readiness_root.expanduser().resolve()
    seal_path = root / "ledger-seal.json"
    observed_seal_sha256 = sha256_file(seal_path)
    if observed_seal_sha256 != FROZEN_READINESS_LEDGER_SEAL_SHA256:
        raise RuntimeError(
            "readiness ledger seal is not the authorized readiness-v2 seal"
        )
    seal = load_canonical_json(seal_path)
    if seal.get("schema_version") != "dense-union-51.final-ledger-seal.v1":
        raise RuntimeError(
            "readiness root does not contain the final Dense-Union-51 seal"
        )
    observed: dict[str, str] = {"ledger-seal.json": observed_seal_sha256}
    artifact_digests = seal.get("artifact_digests")
    source_digests = seal.get("source_digests")
    if not isinstance(artifact_digests, dict) or not isinstance(source_digests, dict):
        raise RuntimeError("readiness ledger seal is incomplete")
    if dict(artifact_digests) != FROZEN_READINESS_ARTIFACT_DIGESTS:
        raise RuntimeError(
            "readiness ledger artifact digest declarations are not authorized"
        )
    if dict(source_digests) != FROZEN_READINESS_SOURCE_DIGESTS:
        raise RuntimeError(
            "readiness ledger source digest declarations are not authorized"
        )
    if seal.get("category_namespace_sha256") != (
        FROZEN_READINESS_CATEGORY_NAMESPACE_SHA256
    ):
        raise RuntimeError("readiness category namespace declaration is not authorized")
    for name, expected in artifact_digests.items():
        path = root / name
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"sealed readiness artifact drifted: {name}")
        observed[name] = actual
    for key, name in _SEALED_SOURCE_FILE_BY_KEY.items():
        expected = source_digests.get(key)
        path = root / name
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"sealed readiness source drifted: {name}")
        observed[name] = actual
    namespace_path = root / "coco-80-category-namespace.json"
    namespace_digest = sha256_file(namespace_path)
    if namespace_digest != seal.get("category_namespace_sha256"):
        raise RuntimeError("Common Objects in Context category namespace drifted")
    observed[namespace_path.name] = namespace_digest

    correction_path = root / "pre-seal-ordering-correction-receipt.json"
    correction = load_canonical_json(correction_path)
    correction_rows = correction.get("artifact_comparisons")
    if not isinstance(correction_rows, list):
        raise RuntimeError("pre-seal ordering correction receipt is incomplete")
    annotation_seal_expected = [
        row.get("new_sha256")
        for row in correction_rows
        if isinstance(row, dict)
        and row.get("name") == "annotation-derived-cohort-seal.json"
    ]
    annotation_seal_path = root / "annotation-derived-cohort-seal.json"
    if annotation_seal_expected != [sha256_file(annotation_seal_path)]:
        raise RuntimeError(
            "annotation-derived cohort seal differs from the final pre-seal correction"
        )
    annotation_seal = load_canonical_json(annotation_seal_path)
    annotation_digests = annotation_seal.get("artifact_digests")
    if not isinstance(annotation_digests, dict):
        raise RuntimeError("annotation-derived cohort seal is incomplete")
    for name, expected in annotation_digests.items():
        path = root / name
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"annotation-derived cohort artifact drifted: {name}")
        observed[name] = actual
    observed[annotation_seal_path.name] = sha256_file(annotation_seal_path)
    return root, observed


def _load_cohort(path: Path) -> CohortLedger:
    return CohortLedger.from_jsonl_bytes(path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-root", type=Path, required=True)
    parser.add_argument("--source-runtime-identity-receipt", type=Path, required=True)
    parser.add_argument("--source-attestation", type=Path, required=True)
    parser.add_argument(
        "--sampled-runtime-attestation-aggregate", type=Path, required=True
    )
    parser.add_argument("--resolved-inference-config", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--calibration-selection-receipt", type=Path, required=True)
    parser.add_argument("--cohort", choices=tuple(_COHORT_FILES), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--root-seed",
        type=int,
        choices=(PRIMARY_ROOT_SEED, SECOND_ROOT_SEED),
        default=PRIMARY_ROOT_SEED,
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    readiness_root, sealed_hashes = _validate_readiness_root(arguments.readiness_root)
    cohort_file_name = _COHORT_FILES[arguments.cohort]
    cohort_path = readiness_root / cohort_file_name
    validation_path = readiness_root / _COHORT_FILES["validation-200"]
    calibration_path = readiness_root / "sampling-calibration-12-manifest.jsonl"
    cohort = _load_cohort(cohort_path)
    validation_cohort = _load_cohort(validation_path)
    calibration_cohort = _load_cohort(calibration_path)
    if cohort.cohort_id != arguments.cohort:
        raise RuntimeError("selected cohort name differs from its manifest identity")

    calibration_selection_path = arguments.calibration_selection_receipt.resolve()
    calibration_selection = CalibrationSelectionReceipt.from_artifact_dict(
        load_canonical_json(calibration_selection_path)
    )
    if (
        calibration_selection.calibration_cohort_sha256
        != calibration_cohort.fingerprint
        or calibration_selection.validation_cohort_sha256
        != validation_cohort.fingerprint
    ):
        raise RuntimeError("calibration selection receipt uses different cohorts")

    identity_path = arguments.source_runtime_identity_receipt.resolve()
    source_runtime_identity = SourceRuntimeIdentityReceipt.from_artifact_dict(
        load_canonical_json(identity_path)
    )
    source_attestation_path = arguments.source_attestation.resolve()
    aggregate_path = arguments.sampled_runtime_attestation_aggregate.resolve()
    config_path = arguments.resolved_inference_config.resolve()
    checkpoint_manifest_path = arguments.checkpoint_manifest.resolve()
    derived_source_runtime_identity = (
        SourceRuntimeIdentityReceipt.from_verified_artifacts(
            source_attestation_path=source_attestation_path,
            sampled_runtime_attestation_aggregate_path=aggregate_path,
            resolved_inference_config_path=config_path,
            checkpoint_manifest_path=checkpoint_manifest_path,
            ledger_seal_sha256=sealed_hashes["ledger-seal.json"],
            processor_contract_sha256=SpatialGridSpec().fingerprint,
        )
    )
    if source_runtime_identity != derived_source_runtime_identity:
        raise RuntimeError(
            "source/runtime identity receipt was not derived from the supplied artifacts"
        )
    policy_set = load_attested_sampling_policy_set(aggregate_path)
    exact_identity_checks = {
        "source attestation": (
            sha256_file(source_attestation_path),
            source_runtime_identity.source_attestation_sha256,
        ),
        "sampled runtime attestation aggregate": (
            sha256_file(aggregate_path),
            source_runtime_identity.sampled_runtime_attestation_aggregate_sha256,
        ),
        "sampled runtime attestation aggregate fingerprint": (
            policy_set.aggregate_payload_fingerprint,
            source_runtime_identity.sampled_runtime_attestation_aggregate_fingerprint,
        ),
        "resolved inference config": (
            sha256_file(config_path),
            source_runtime_identity.config_sha256,
        ),
        "checkpoint manifest": (
            sha256_file(checkpoint_manifest_path),
            source_runtime_identity.checkpoint_manifest_sha256,
        ),
        "readiness ledger seal": (
            sealed_hashes["ledger-seal.json"],
            source_runtime_identity.ledger_seal_sha256,
        ),
    }
    mismatches = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in exact_identity_checks.items()
        if actual != expected
    }
    if mismatches:
        raise RuntimeError(f"source/runtime identity mismatch: {mismatches}")
    if policy_set != calibration_selection.attested_policy_set:
        raise RuntimeError("selected policy receipt uses another attestation aggregate")

    source_hashes: dict[str, str] = {
        **sealed_hashes,
        cohort_file_name: sha256_file(cohort_path),
        "calibration-selection-receipt.json": sha256_file(calibration_selection_path),
        "source-runtime-identity-receipt.json": sha256_file(identity_path),
        "sampling-calibration-12-manifest.jsonl": sha256_file(calibration_path),
        "validation_cohort_fingerprint": validation_cohort.fingerprint,
        "calibration_cohort_fingerprint": calibration_cohort.fingerprint,
        "source-attestation.json": sha256_file(source_attestation_path),
        "sampled-runtime-attestation-aggregate.json": sha256_file(aggregate_path),
        "resolved-inference-config": sha256_file(config_path),
        "checkpoint-manifest.json": sha256_file(checkpoint_manifest_path),
    }
    artifact = materialize_primary_schedule_artifact(
        unit_id=UNIT_ID,
        run_id=arguments.run_id,
        cohort=cohort,
        cohort_artifact_name=cohort_file_name,
        cohort_artifact_sha256=sha256_file(cohort_path),
        readiness_ledger_seal_sha256=sealed_hashes["ledger-seal.json"],
        calibration_selection=calibration_selection,
        calibration_selection_receipt_sha256=sha256_file(calibration_selection_path),
        source_runtime_identity=source_runtime_identity,
        source_runtime_identity_receipt_sha256=sha256_file(identity_path),
        source_hashes=source_hashes,
        root_seed=arguments.root_seed,
    )
    write_immutable_json(arguments.output, artifact.to_artifact_dict())
    summary: dict[str, Any] = {
        "artifact_sha256": artifact.artifact_sha256,
        "cohort_id": cohort.cohort_id,
        "physical_batch_count": len(artifact.schedule.batches()),
        "request_count": len(artifact.schedule.requests),
        "run_id": artifact.schedule.identity.run_id,
        "schedule_sha256": artifact.schedule.fingerprint,
        "selected_temperature": artifact.schedule.identity.decode.temperature,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
