from __future__ import annotations

from pathlib import Path

from repo_lifecycle.report_lifecycle_registry import DEFAULT_REGISTRY, build_report


def test_refactoring_lifecycle_registry_is_well_formed() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report = build_report(repo_root=repo_root, registry_path=repo_root / DEFAULT_REGISTRY)

    assert report.errors == []
    assert report.status_counts["active-research"] >= 1
    assert report.status_counts["preserved-comparator"] >= 1
    assert report.status_counts["needs-classification"] >= 1
