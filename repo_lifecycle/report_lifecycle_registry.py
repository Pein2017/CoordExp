#!/usr/bin/env python
"""Report lifecycle-registry coverage for the refactoring program.

The default mode is intentionally report-only: malformed registry entries and
missing required paths are errors, while broad lifecycle findings are printed as
warnings. Use ``--strict`` once the initial findings have been classified.

This module intentionally lives outside ``src/`` and ``scripts/``: it is
repository-governance machinery, not CoordExp runtime code or an experiment
entrypoint.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY = Path(
    "docs/architecture/proposals/2026-06-17-refactoring-program/lifecycle_registry.yaml"
)


@dataclass
class Finding:
    level: str
    message: str


@dataclass
class RegistryReport:
    errors: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    status_counts: dict[str, int] = field(default_factory=dict)


def _repo_root_from(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Could not find repo root from {start}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _text_files_under(repo_root: Path, include: list[str], allow: list[str]) -> list[Path]:
    allow_paths = [(repo_root / item).resolve() for item in allow]
    files: list[Path] = []
    for item in include:
        root = repo_root / item
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = [path for path in root.rglob("*") if path.is_file()]
        for path in candidates:
            resolved = path.resolve()
            if any(_is_under(resolved, allowed) for allowed in allow_paths):
                continue
            if ".git" in path.parts or ".codegraph" in path.parts:
                continue
            files.append(path)
    return files


def build_report(repo_root: Path, registry_path: Path) -> RegistryReport:
    report = RegistryReport()
    registry = _load_yaml(registry_path)

    taxonomy = registry.get("status_taxonomy")
    if not isinstance(taxonomy, dict):
        report.errors.append("status_taxonomy must be a mapping")
        allowed_statuses: set[str] = set()
    else:
        allowed_statuses = set(taxonomy)

    entries = registry.get("entries")
    if not isinstance(entries, list):
        report.errors.append("entries must be a list")
        return report

    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            report.errors.append(f"entry #{index} must be a mapping")
            continue
        entry_id = entry.get("id")
        status = entry.get("status")
        paths = entry.get("paths")
        if not isinstance(entry_id, str) or not entry_id:
            report.errors.append(f"entry #{index} has missing/invalid id")
            continue
        if entry_id in seen_ids:
            report.errors.append(f"duplicate entry id: {entry_id}")
        seen_ids.add(entry_id)
        if status not in allowed_statuses:
            report.errors.append(f"{entry_id}: unknown status {status!r}")
        else:
            report.status_counts[status] = report.status_counts.get(status, 0) + 1
        if not isinstance(paths, list) or not paths:
            report.errors.append(f"{entry_id}: paths must be a non-empty list")
            continue
        for raw_path in paths:
            if not isinstance(raw_path, str) or not raw_path:
                report.errors.append(f"{entry_id}: invalid path {raw_path!r}")
                continue
            path = repo_root / raw_path
            if not path.exists():
                report.errors.append(f"{entry_id}: missing path {raw_path}")

    checks = registry.get("report_checks") or {}
    if not isinstance(checks, dict):
        report.errors.append("report_checks must be a mapping when present")
        return report

    for root_check in checks.get("unclassified_roots", []) or []:
        if not isinstance(root_check, dict):
            report.errors.append("unclassified_roots entries must be mappings")
            continue
        raw_root = root_check.get("root")
        covered_by = root_check.get("covered_by")
        if not isinstance(raw_root, str) or not isinstance(covered_by, str):
            report.errors.append(f"invalid unclassified root check: {root_check!r}")
            continue
        root = repo_root / raw_root
        if not root.exists():
            report.findings.append(Finding("warn", f"unclassified root missing: {raw_root}"))
            continue
        families = sorted(
            child.name for child in root.iterdir() if not child.name.startswith(".")
        )
        report.findings.append(
            Finding(
                "info",
                f"{raw_root}: {len(families)} first-level entries covered by placeholder {covered_by}",
            )
        )

    for pattern_check in checks.get("text_patterns", []) or []:
        if not isinstance(pattern_check, dict):
            report.errors.append("text_patterns entries must be mappings")
            continue
        check_id = pattern_check.get("id")
        pattern = pattern_check.get("pattern")
        include = pattern_check.get("include", [])
        allow = pattern_check.get("allow", [])
        if not isinstance(check_id, str) or not isinstance(pattern, str):
            report.errors.append(f"invalid text pattern check: {pattern_check!r}")
            continue
        if not isinstance(include, list) or not all(isinstance(item, str) for item in include):
            report.errors.append(f"{check_id}: include must be a list of strings")
            continue
        if not isinstance(allow, list) or not all(isinstance(item, str) for item in allow):
            report.errors.append(f"{check_id}: allow must be a list of strings")
            continue
        regex = re.compile(pattern)
        matches: list[str] = []
        for path in _text_files_under(repo_root, include, allow):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{path.relative_to(repo_root)}:{line_no}")
                    if len(matches) >= 20:
                        break
            if len(matches) >= 20:
                break
        if matches:
            report.findings.append(
                Finding("warn", f"{check_id}: {len(matches)} sample matches: {', '.join(matches)}")
            )
        else:
            report.findings.append(Finding("info", f"{check_id}: no matches"))

    return report


def print_report(report: RegistryReport) -> None:
    print("Lifecycle registry report")
    print("=========================")
    if report.status_counts:
        print("Status counts:")
        for status in sorted(report.status_counts):
            print(f"  {status}: {report.status_counts[status]}")
    if report.findings:
        print("Findings:")
        for finding in report.findings:
            print(f"  [{finding.level}] {finding.message}")
    if report.errors:
        print("Errors:")
        for error in report.errors:
            print(f"  [error] {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warnings too")
    args = parser.parse_args(argv)

    repo_root = args.repo_root or _repo_root_from(Path.cwd())
    registry_path = args.registry
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path
    report = build_report(repo_root=repo_root, registry_path=registry_path)
    print_report(report)
    if report.errors:
        return 2
    if args.strict and any(finding.level == "warn" for finding in report.findings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
