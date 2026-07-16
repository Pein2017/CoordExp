#!/usr/bin/env python3
"""Materialize one committed COCO refinement split to its fixed coord output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.label_studio_coco_refinement.materialize import (  # noqa: E402
    ProjectTaskIndexSourceIdentityResolver,
    WorkingCoordMaterializer,
)
from src.label_studio_coco_refinement.models import (  # noqa: E402
    RefinementRuntimeLayout,
)
from src.label_studio_coco_refinement.store import (  # noqa: E402
    WorkingDatasetStore,
)


class _UnavailableAnnotationVerifier:
    def verify(self, identity: Any) -> bool:
        raise RuntimeError("materialization must not consult live annotations")


class _UnavailableInferenceResolver:
    def resolve(self, receipt_id: str) -> None:
        raise RuntimeError("materialization must not consult inference receipts")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export the exact reconciled working.norm.jsonl generation to the "
            "fixed working.coord.jsonl sibling and durable receipt."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="CoordExp repository root owning the selected source and runtime.",
    )
    parser.add_argument("--split", choices=("train", "val"), required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo_root = args.repo_root.expanduser().resolve(strict=True)
    if not repo_root.is_dir():
        raise SystemExit(f"repository root is not a directory: {repo_root}")
    layout = RefinementRuntimeLayout.under_repository(repo_root)
    store = WorkingDatasetStore(
        layout.split_root(args.split),
        annotation_verifier=_UnavailableAnnotationVerifier(),
        inference_receipt_resolver=_UnavailableInferenceResolver(),
        recover=False,
    )
    resolver = ProjectTaskIndexSourceIdentityResolver(
        layout,
        args.split,
        store=store,
    )
    receipt = WorkingCoordMaterializer(
        layout,
        args.split,
        store=store,
        source_identity_resolver=resolver,
    ).materialize_current_generation()
    print(
        json.dumps(
            receipt.to_artifact_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
