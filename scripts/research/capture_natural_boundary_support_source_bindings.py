#!/usr/bin/env python3
"""Capture the exact source inputs for the resumable support-shard change.

This is a provenance-only boundary.  It reads source bytes and the active
``research-probes`` Git metadata; it never edits, stages, commits, or
otherwise mutates the active research unit.  The resulting document is strict
canonical JSON and is published write-once.  ``sources`` is the stable
inventory (resolved ``path``, ``byte_count``, ``sha256``, and Git provenance),
``git.head`` is stable for a fixed checkout, and ``git.status`` is the only
explicitly volatile section.  ``stable_bindings_sha256`` excludes that status
section; ``self_sha256`` covers the complete published document.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


# Keep direct ``python scripts/research/...`` invocation independent of the
# caller's working directory.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.artifacts.json_values import canonical_json_bytes, json_sha256, publish_json_exclusive  # noqa: E402
from src.common.errors import ArtifactContractError  # noqa: E402


UNIT_ID = "2026-08-06-natural-boundary-routing-history-replication"
SCHEMA_VERSION = "natural_boundary_support_source_bindings.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_RESEARCH_PROBES_ROOT = REPO_ROOT.parent / "research-probes"
OUTPUT_PATH = (
    REPO_ROOT
    / "openspec/changes/add-resumable-costed-research-probe-shards/verification/source-bindings.json"
)
OUTPUT = OUTPUT_PATH

class SourceBindingError(ValueError):
    """Raised when a source cannot be proven and bound safely."""


# Friendly compatibility aliases for callers that prefer a receipt-specific
# name.  The public error remains a plain ValueError subclass.
CaptureError = SourceBindingError
SourceBindingsError = SourceBindingError


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of raw bytes."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one validated regular non-symlink file."""

    return _read_regular_file(path, label="source")[1]["sha256"]


def _strict_text(value: Any, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        raise SourceBindingError(f"{label} must be a non-empty string")
    return value


def _strict_sha256(value: Any, label: str) -> str:
    text = _strict_text(value, label).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise SourceBindingError(f"{label} must be a lowercase SHA-256")
    return text


def _read_regular_file(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    """Read one source exactly once, refusing symlinks and non-files."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise SourceBindingError(f"{label} must not be a symlink: {candidate}")
    if not candidate.exists():
        raise SourceBindingError(f"{label} does not exist: {candidate}")
    if not candidate.is_file():
        raise SourceBindingError(f"{label} is not a regular file: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SourceBindingError(f"{label} could not be resolved: {candidate}") from exc
    # ``candidate`` was checked before resolution; check the resolved path too
    # so a race cannot silently turn the bound object into a symlink.
    if resolved.is_symlink() or not resolved.is_file():
        raise SourceBindingError(f"{label} is not a regular non-symlink file: {candidate}")
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise SourceBindingError(f"{label} is unreadable: {resolved}") from exc
    info = {
        "path": str(resolved),
        "byte_count": len(raw),
        "sha256": sha256_bytes(raw),
    }
    return resolved, info


def _run_git(root: Path, args: Sequence[str], *, label: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SourceBindingError(f"could not inspect Git {label} under {root}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SourceBindingError(f"Git {label} failed under {root}: {detail}")
    return result.stdout


def _git_head(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise SourceBindingError(f"active Git root is not a directory: {root}")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise SourceBindingError(f"active Git root could not be resolved: {root}") from exc
    commit = _run_git(resolved, ["rev-parse", "HEAD"], label="HEAD").strip()
    if not commit:
        raise SourceBindingError(f"Git HEAD is empty under {resolved}")
    branch_result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=resolved,
        check=False,
        capture_output=True,
        text=True,
    )
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None
    return {
        "commit": commit,
        "branch": branch,
        "detached": branch is None,
    }


def _git_status(root: Path) -> dict[str, Any]:
    raw = _run_git(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        label="status",
    )
    entries = raw.splitlines()
    # Status is the only intentionally volatile section.  Do not add a time
    # stamp: two captures with unchanged inputs should remain byte-identical.
    return {
        "volatile": True,
        "format": "porcelain-v1",
        "entries": entries,
        "entry_count": len(entries),
    }


def _relative_to_root(path: Path, root: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _git_source_provenance(path: Path, *, root: Path, head: Mapping[str, Any]) -> dict[str, Any]:
    """Prove tracked/untracked state without assigning ownership to untracked bytes."""

    relative = _relative_to_root(path, root)
    if relative is None:
        return {
            "state": "external",
            "tracked": False,
            "owner_commit": None,
            "commit": None,
            "proof": "outside_active_research_probes_root",
        }

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0
    status_result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if status_result.returncode != 0:
        raise SourceBindingError(f"Git status could not prove source state: {path}")
    status_lines = status_result.stdout.splitlines()
    if tracked:
        state = "modified" if status_lines else "tracked_clean"
        owner_commit: str | None = str(head["commit"])
        proof = "git_ls_files_and_status"
    elif any(line.startswith("??") for line in status_lines):
        state = "untracked"
        owner_commit = None
        proof = "git_status_porcelain_untracked"
    else:
        ignored_result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if ignored_result.returncode == 0:
            state = "ignored"
            proof = "git_check_ignore"
        else:
            # A source can be outside status output when the repository has
            # unusual index/configuration rules.  Keep the state honest rather
            # than fabricating a commit identity.
            state = "untracked_unproven"
            proof = "git_ls_files_absent_status_empty"
        owner_commit = None
    return {
        "state": state,
        "tracked": tracked,
        "owner_commit": owner_commit,
        "commit": owner_commit,
        "proof": proof,
    }


def _bind_file(
    path: str | Path,
    *,
    label: str,
    active_root: Path,
    head: Mapping[str, Any],
) -> dict[str, Any]:
    resolved, info = _read_regular_file(path, label=label)
    return {
        "path": info["path"],
        "byte_count": info["byte_count"],
        "sha256": info["sha256"],
        "git": _git_source_provenance(resolved, root=active_root, head=head),
    }


def _default_paths(active_root: Path) -> dict[str, Path]:
    return {
        "consumer": active_root / "scripts/research/run_natural_boundary_support_completion.py",
        "merger": active_root / "scripts/research/merge_natural_boundary_support_completion.py",
        "materializer": active_root / "scripts/research/materialize_natural_boundary_census_v3.py",
        "analyzer": active_root / "scripts/research/analyze_s_natural_boundary_k_n_h_evidence.py",
        "consumer_test": active_root / "tests/research/test_run_natural_boundary_support_completion.py",
        "merger_test": active_root / "tests/test_merge_natural_boundary_support_completion.py",
        "materializer_test": active_root / "tests/test_natural_boundary_census_v3.py",
        "analyzer_test": active_root / "tests/research/test_analyze_s_natural_boundary_k_n_h_evidence.py",
        "sealed_plan": Path(
            "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
            f"{UNIT_ID}/support-completion-plan-v1/plan.json"
        ),
        "planner_receipt": Path(
            "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
            f"{UNIT_ID}/support-completion-plan-v1/receipt.json"
        ),
        "census": Path(
            "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
            f"{UNIT_ID}/cpu-census-v2/admission-census.json"
        ),
    }


def _coerce_path(value: str | Path | None, default: Path) -> Path:
    return default if value is None else Path(value).expanduser()


def capture_source_bindings(
    *,
    active_root: str | Path = ACTIVE_RESEARCH_PROBES_ROOT,
    consumer: str | Path | None = None,
    merger: str | Path | None = None,
    materializer: str | Path | None = None,
    analyzer: str | Path | None = None,
    consumer_test: str | Path | None = None,
    merger_test: str | Path | None = None,
    materializer_test: str | Path | None = None,
    analyzer_test: str | Path | None = None,
    sealed_plan: str | Path | None = None,
    planner_receipt: str | Path | None = None,
    census: str | Path | None = None,
) -> dict[str, Any]:
    """Capture the fixed source inventory and active Git state.

    Every argument is read-only.  The returned mapping is built entirely from
    plain JSON values and can therefore be passed directly to
    :func:`write_source_bindings`.
    """

    root_candidate = Path(active_root).expanduser()
    if root_candidate.is_symlink() or not root_candidate.is_dir():
        raise SourceBindingError(f"active research-probes root is not a directory: {root_candidate}")
    root = root_candidate.resolve(strict=True)
    head = _git_head(root)
    status = _git_status(root)
    defaults = _default_paths(root)

    paths = {
        "consumer": _coerce_path(consumer, defaults["consumer"]),
        "merger": _coerce_path(merger, defaults["merger"]),
        "materializer": _coerce_path(materializer, defaults["materializer"]),
        "analyzer": _coerce_path(analyzer, defaults["analyzer"]),
        "consumer_test": _coerce_path(consumer_test, defaults["consumer_test"]),
        "merger_test": _coerce_path(merger_test, defaults["merger_test"]),
        "materializer_test": _coerce_path(materializer_test, defaults["materializer_test"]),
        "analyzer_test": _coerce_path(analyzer_test, defaults["analyzer_test"]),
        "sealed_plan": _coerce_path(sealed_plan, defaults["sealed_plan"]),
        "planner_receipt": _coerce_path(planner_receipt, defaults["planner_receipt"]),
        "census": _coerce_path(census, defaults["census"]),
    }

    files = {
        name: _bind_file(path, label=name.replace("_", " "), active_root=root, head=head)
        for name, path in paths.items()
    }
    sources = {
        "consumer": files["consumer"],
        "merger": files["merger"],
        "direct_analyzers": {
            "materializer": files["materializer"],
            "analyzer": files["analyzer"],
        },
        "tests": {
            "consumer": files["consumer_test"],
            "merger": files["merger_test"],
            "materializer": files["materializer_test"],
            "analyzer": files["analyzer_test"],
        },
        "sealed_plan": files["sealed_plan"],
        "planner_receipt": files["planner_receipt"],
        "census": files["census"],
    }

    # The two active consumer files may be tracked after integration or
    # untracked while they are still owned by a sibling worktree.  In either
    # state, the receipt must never invent a commit owner for untracked bytes.
    for role in ("consumer", "merger"):
        provenance = files[role]["git"]
        if provenance["state"] == "untracked" and provenance["owner_commit"] is not None:
            raise SourceBindingError(f"{role} is untracked but has an owner commit")

    stable = {"sources": sources, "git_head": head}
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "captured",
        "unit_id": UNIT_ID,
        "claim_boundary": {
            "scientific_claims": False,
            "text": (
                "This receipt binds source bytes and Git provenance only; it is not "
                "model, mechanism, support-prevalence, execution, or deployment evidence."
            ),
        },
        "active_unit_non_mutation": {
            "intent": (
                "Read the active research-probes unit without editing, staging, committing, "
                "merging, cherry-picking, pushing, or rewriting sealed sources."
            ),
            "sources_edited": False,
            "sources_staged": False,
            "sources_committed": False,
            "active_unit_mutated": False,
        },
        "sources": sources,
        "git": {
            "active_root": str(root),
            "head": head,
            "status": status,
        },
        "stable_bindings_sha256": json_sha256(stable),
    }
    document["self_sha256"] = json_sha256(document)
    return document


# API aliases used by lightweight callers and focused tests.
build_source_bindings = capture_source_bindings
capture_bindings = capture_source_bindings


def stable_bindings(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic portion used for receipt comparisons."""

    if not isinstance(document, Mapping):
        raise SourceBindingError("source-binding receipt must be a mapping")
    sources = document.get("sources")
    git = document.get("git")
    if not isinstance(sources, Mapping) or not isinstance(git, Mapping) or not isinstance(git.get("head"), Mapping):
        raise SourceBindingError("source-binding receipt lacks stable source/head sections")
    return {"sources": dict(sources), "git_head": dict(git["head"])}


stable_source_bindings = stable_bindings


def validate_source_bindings(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate canonical receipt shape and recompute both hashes."""

    if not isinstance(document, Mapping):
        raise SourceBindingError("source-binding receipt must be a mapping")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("status") != "captured":
        raise SourceBindingError("source-binding receipt schema/status mismatch")
    _strict_text(document.get("unit_id"), "unit_id")
    claim = document.get("claim_boundary")
    if not isinstance(claim, Mapping) or claim.get("scientific_claims") is not False:
        raise SourceBindingError("source-binding receipt claim boundary is not mechanics-only")
    non_mutation = document.get("active_unit_non_mutation")
    if not isinstance(non_mutation, Mapping) or non_mutation.get("active_unit_mutated") is not False:
        raise SourceBindingError("source-binding receipt lacks active-unit non-mutation intent")
    stable = stable_bindings(document)
    if document.get("stable_bindings_sha256") != json_sha256(stable):
        raise SourceBindingError("stable source-binding digest mismatch")
    self_hash = document.get("self_sha256")
    _strict_sha256(self_hash, "self_sha256")
    body = dict(document)
    body.pop("self_sha256", None)
    if json_sha256(body) != self_hash:
        raise SourceBindingError("source-binding self digest mismatch")
    return dict(document)


def _publish_write_once(path: Path, payload: bytes) -> Path:
    """Publish canonical bytes atomically, accepting only an identical retry."""

    destination = Path(path).expanduser()
    if destination.is_symlink():
        raise SourceBindingError(f"output must not be a symlink: {destination}")
    if destination.exists():
        if not destination.is_file():
            raise SourceBindingError(f"output is not a regular file: {destination}")
        try:
            existing = destination.read_bytes()
        except OSError as exc:
            raise SourceBindingError(f"output is unreadable: {destination}") from exc
        if existing != payload:
            raise SourceBindingError(f"refusing to overwrite differing source-binding bytes: {destination}")
        return destination.resolve(strict=True)
    try:
        publish_json_exclusive(destination, json.loads(payload.decode("utf-8")))
    except ArtifactContractError as exc:
        # A concurrent identical publication is a successful idempotent retry;
        # a differing publication remains an explicit refusal.
        if destination.is_file() and not destination.is_symlink():
            existing = destination.read_bytes()
            if existing == payload:
                return destination.resolve(strict=True)
        raise SourceBindingError(str(exc)) from exc
    return destination.resolve(strict=True)


def write_source_bindings(
    output: str | Path = OUTPUT_PATH,
    *,
    document: Mapping[str, Any] | None = None,
    **capture_kwargs: Any,
) -> dict[str, Any]:
    """Capture (unless supplied) and publish one canonical write-once receipt."""

    payload_document = capture_source_bindings(**capture_kwargs) if document is None else dict(document)
    validate_source_bindings(payload_document)
    encoded = canonical_json_bytes(payload_document)
    _publish_write_once(Path(output), encoded)
    return payload_document


publish_source_bindings = write_source_bindings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--active-root", type=Path, default=ACTIVE_RESEARCH_PROBES_ROOT)
    for name in (
        "consumer",
        "merger",
        "materializer",
        "analyzer",
        "consumer-test",
        "merger-test",
        "materializer-test",
        "analyzer-test",
        "sealed-plan",
        "planner-receipt",
        "census",
    ):
        parser.add_argument(f"--{name}", dest=name.replace("-", "_"), type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kwargs = {
        key: getattr(args, key)
        for key in (
            "consumer",
            "merger",
            "materializer",
            "analyzer",
            "consumer_test",
            "merger_test",
            "materializer_test",
            "analyzer_test",
            "sealed_plan",
            "planner_receipt",
            "census",
        )
        if getattr(args, key) is not None
    }
    try:
        document = write_source_bindings(args.output, active_root=args.active_root, **kwargs)
    except (SourceBindingError, ArtifactContractError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"capture-natural-boundary-source-bindings: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
