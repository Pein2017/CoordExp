#!/usr/bin/env python3
"""Compile explicit adjudicator decisions into immutable final ledger artifacts."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.spatial_scope_history import (  # noqa: E402
    assemble_final_review_ledger,
    build_bound_adjudication_queue,
)
from src.analysis.spatial_scope_history.cohort_ledger import (  # noqa: E402
    canonical_json_text,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "review-queue",
        "reviewer-one-labels",
        "reviewer-two-labels",
        "official-individual-ledger",
        "official-crowd-ledger",
        "pre-seal-ordering-correction-receipt",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "expected-review-queue-sha256",
        "expected-packet-sha256",
        "expected-ontology-sha256",
        "expected-pre-seal-ordering-correction-receipt-sha256",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--adjudication-queue", type=Path)
    parser.add_argument("--adjudicator-decisions", type=Path)
    parser.add_argument("--ledger-version")
    parser.add_argument("--seal-created-at")
    parser.add_argument("--earliest-permitted-metric-run-start")
    parser.add_argument("--build-queue-only", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    common = {
        "review_queue_jsonl": args.review_queue.read_bytes(),
        "reviewer_one_labels_jsonl": args.reviewer_one_labels.read_bytes(),
        "reviewer_two_labels_jsonl": args.reviewer_two_labels.read_bytes(),
        "official_individual_ledger_jsonl": args.official_individual_ledger.read_bytes(),
        "official_crowd_ledger_jsonl": args.official_crowd_ledger.read_bytes(),
        "pre_seal_ordering_correction_receipt_json": (
            args.pre_seal_ordering_correction_receipt.read_bytes()
        ),
        "expected_review_queue_sha256": args.expected_review_queue_sha256,
        "expected_packet_sha256": args.expected_packet_sha256,
        "expected_ontology_sha256": args.expected_ontology_sha256,
        "expected_pre_seal_ordering_correction_receipt_sha256": (
            args.expected_pre_seal_ordering_correction_receipt_sha256
        ),
    }
    if args.build_queue_only:
        queue = build_bound_adjudication_queue(**common)
        summary = {
            "adjudication_queue_row_count": len(queue.splitlines()),
            "adjudication_queue_sha256": hashlib.sha256(queue).hexdigest(),
            "artifact_role": "decision-free adjudication queue materialization",
            "pre_seal_ordering_correction_receipt_sha256": (
                args.expected_pre_seal_ordering_correction_receipt_sha256
            ),
            "schema_version": "dense-union-51.adjudication-queue-materialization.v1",
        }
        _publish_once(
            args.output_root,
            {
                "adjudication-queue.jsonl": queue,
                "adjudication-queue-summary.json": (
                    canonical_json_text(summary) + "\n"
                ).encode(),
            },
        )
        return
    final_fields = (
        "adjudication_queue",
        "adjudicator_decisions",
        "ledger_version",
        "seal_created_at",
        "earliest_permitted_metric_run_start",
    )
    missing = [name for name in final_fields if getattr(args, name) is None]
    if missing:
        parser.error(f"final assembly requires: {', '.join(missing)}")
    artifacts = assemble_final_review_ledger(
        **common,
        adjudication_queue_jsonl=args.adjudication_queue.read_bytes(),
        adjudicator_decisions_jsonl=args.adjudicator_decisions.read_bytes(),
        ledger_version=args.ledger_version,
        seal_created_at=args.seal_created_at,
        earliest_permitted_metric_run_start=args.earliest_permitted_metric_run_start,
    )
    _publish_once(
        args.output_root,
        {
            "adjudication.jsonl": artifacts.adjudication_jsonl,
            "audit-augmented-ledger.jsonl": artifacts.audit_augmented_ledger_jsonl,
            "ledger-seal.json": artifacts.ledger_seal_json,
        },
    )


def _publish_once(root: Path, artifacts: dict[str, bytes]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    targets = {name: root / name for name in artifacts}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to replace final artifact(s): {existing}")
    temporary: list[Path] = []
    published: list[Path] = []
    try:
        for name, payload in artifacts.items():
            path = root / f".{name}.{os.getpid()}.tmp"
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.append(path)
        for path, target in zip(temporary, targets.values(), strict=True):
            os.link(path, target)
            published.append(target)
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
