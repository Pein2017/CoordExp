#!/usr/bin/env python3
"""Read-only checks for the agent research map and its frozen source archive.

This checks local structure and provenance, not scientific claims or external
artifacts. Historical links use their original logical document coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import posixpath
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

PROGRAM = Path("research/qwen3-vl-dense-enumeration")
CAPTURE = Path("docs/history/research-records/2026-09-15")
BASELINE = "78bc27d8e5e8639450e01990908fcd3b417bbfa2"
LIFECYCLES = {"planned", "ready", "running", "blocked", "paused", "closed", "superseded"}
EVIDENCE_STATES = {"none", "partial", "unreviewed", "accepted", "invalid"}
LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
FENCED = re.compile(r"(?ms)^\s*```[^\n]*\n.*?^\s*```[^\n]*$")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def local_path(root: Path, name: str) -> Path:
    """Reject escaping paths before reading any bytes; legacy in-root aliases work."""
    if not isinstance(name, str) or not name or any(c in name for c in "\x00\n\r"):
        raise ValueError("invalid repository path")
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"not a repository-relative path: {name}")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path resolves outside repository: {name}")
    return result


def logical_relative(root: Path, name: str, manifest: dict[str, Any]) -> str | None:
    if name.startswith("/"):
        for prefix in (str(root.resolve()), manifest.get("canonical_root", "")):
            if prefix and (name == prefix or name.startswith(prefix.rstrip("/") + "/")):
                name = name[len(prefix):].lstrip("/")
                break
        else:
            return None
    value = posixpath.normpath(name)
    return None if value == ".." or value.startswith("../") else value


def resolve_reference(root: Path, manifest: dict[str, Any], document: str,
                      target: str) -> dict[str, Any]:
    """Resolve a local reference without probing files outside this checkout."""
    parts = urlsplit(target.strip().strip("<>"))
    if parts.scheme or parts.netloc:
        return {"kind": "external", "target": target, "exists": None}
    document_rel = logical_relative(root, document, manifest)
    if document_rel is None:
        raise ValueError("document is outside the registered checkout")
    reverse = {e["archive"]: e["source"] for e in manifest["files"]}
    prefix = manifest["source_prefix"]
    historical = (document_rel in reverse or document_rel == prefix
                  or document_rel.startswith(prefix + "/"))
    source = reverse.get(document_rel, document_rel)
    link_path = unquote(parts.path)
    candidate = (link_path if link_path.startswith("/") else
                 posixpath.join(posixpath.dirname(source), link_path)) if link_path else source
    logical = logical_relative(root, candidate, manifest)
    if logical is None:
        kind = "external" if link_path.startswith("/") else "invalid"
        return {"kind": kind, "target": target, "exists": None}
    # A live backlink must not be hijacked by a before-edit router snapshot.
    source_map = {e["source"]: e["archive"] for e in manifest["files"]
                  if historical or e["action"] == "relocated_byte_exact"}
    resolved = source_map.get(logical, logical)
    if logical == prefix or logical.startswith(prefix + "/"):
        resolved = manifest["archive_prefix"] + logical[len(prefix):]
    try:
        path = local_path(root, resolved)
    except ValueError:
        return {"kind": "invalid", "logical_path": logical, "target": target, "exists": None}
    return {"kind": "local", "logical_path": logical, "path": resolved,
            "fragment": parts.fragment, "exists": path.exists()}


def check_sources(root: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    seen_source: set[str] = set()
    seen_archive: set[str] = set()
    for entry in manifest["files"]:
        source, archived = entry["source"], entry["archive"]
        if source in seen_source or archived in seen_archive:
            errors.append(f"duplicate manifest mapping: {source}")
        seen_source.add(source)
        seen_archive.add(archived)
        try:
            data = local_path(root, archived).read_bytes()
            if digest(data) != entry["sha256"] or len(data) != entry["bytes"]:
                errors.append(f"source bytes changed: {archived}")
        except (OSError, ValueError) as exc:
            errors.append(f"source unavailable: {archived}: {exc}")
    return errors


def check_git_sources(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Verify tracked originals against Git, not merely a self-consistent manifest."""
    if manifest.get("baseline_head") != BASELINE:
        return ["unexpected archive baseline identity"]
    originals = [e for e in manifest["files"] if e["git_tracked_at_capture"]]
    tree = subprocess.run(
        ["git", "--no-replace-objects", "ls-tree", "-r", "--name-only", BASELINE,
         "--", manifest["source_prefix"]], cwd=root, check=True,
        capture_output=True, text=True, timeout=30).stdout.splitlines()
    expected = {e["source"] for e in originals if e["action"] == "relocated_byte_exact"}
    errors = [] if set(tree) == expected else ["manifest omits/adds original tracked source paths"]
    request = "".join(f"{BASELINE}:{e['source']}\n" for e in originals).encode()
    process = subprocess.run(["git", "--no-replace-objects", "cat-file", "--batch"],
                             cwd=root, input=request, capture_output=True, check=True,
                             timeout=30)
    stream = io.BytesIO(process.stdout)
    for entry in originals:
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError(f"missing original Git blob: {entry['source']}")
        data = stream.read(int(header[2]))
        if stream.read(1) != b"\n":
            raise ValueError("invalid Git batch framing")
        if digest(data) != entry["sha256"]:
            errors.append(f"manifest differs from original Git bytes: {entry['source']}")
    return errors


def check_state(root: Path, state: dict[str, Any], unit_id: str) -> list[str]:
    errors: list[str] = []
    required = {"schema_version", "unit_id", "lifecycle", "evidence", "disposition",
                "state_as_of", "protocol", "result", "state_source", "boundary",
                "not_authorized", "next_action"}
    if not required.issubset(state):
        errors.append(f"{unit_id}: missing state fields {sorted(required - state.keys())}")
    if state.get("schema_version") != 1 or state.get("unit_id") != unit_id:
        errors.append(f"{unit_id}: state identity/schema mismatch")
    if state.get("lifecycle") not in LIFECYCLES:
        errors.append(f"{unit_id}: invalid lifecycle")
    if state.get("evidence") not in EVIDENCE_STATES:
        errors.append(f"{unit_id}: invalid evidence axis")
    for key in ("disposition", "state_as_of", "boundary", "next_action"):
        if not isinstance(state.get(key), str) or not state[key].strip():
            errors.append(f"{unit_id}: empty {key}")
    if not isinstance(state.get("not_authorized"), list):
        errors.append(f"{unit_id}: not_authorized must be a list")
    if state.get("evidence") == "accepted" and not state.get("result"):
        errors.append(f"{unit_id}: accepted evidence lacks a result owner")
    for key in ("protocol", "result", "state_source"):
        name = state.get(key)
        if key == "result" and name is None:
            continue
        try:
            if not local_path(root, name).is_file():
                errors.append(f"{unit_id}: missing {key}: {name}")
        except (TypeError, ValueError) as exc:
            errors.append(f"{unit_id}: invalid {key}: {exc}")
    return errors


def check_catalog(root: Path, rows: list[dict[str, Any]], manifest: dict[str, Any],
                  program: Path = PROGRAM) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    protocols: set[str] = set()
    states: set[str] = set()
    for row in rows:
        unit_id = row["id"]
        if unit_id in ids:
            errors.append(f"duplicate catalog id: {unit_id}")
        ids.add(unit_id)
        if not row.get("topics"):
            errors.append(f"{unit_id}: no question retrieval tag")
        for topic in row.get("topics", []):
            if not local_path(root, str(program / "questions" / f"{topic}.md")).is_file():
                errors.append(f"{unit_id}: missing question page {topic}")
        names = [row["record_root"], row["reading_entry"], *row["protocols"],
                 *row["result_records"]]
        for name in names:
            if not local_path(root, name).exists():
                errors.append(f"{unit_id}: unresolved catalog path {name}")
        protocols.update(row["protocols"])
        if row.get("tracking") not in {"historical", "current"}:
            errors.append(f"{unit_id}: invalid tracking role")
        if row.get("tracking") == "current" and not row.get("state"):
            errors.append(f"{unit_id}: current entry has no state owner")
        if row.get("tracking") == "current" and "handoff" in Path(row["reading_entry"]).name:
            errors.append(f"{unit_id}: handoff cannot own the live route")
        if row.get("state"):
            state_path = row["state"]
            if state_path in states:
                errors.append(f"duplicate state owner: {state_path}")
            states.add(state_path)
            state = json.loads(local_path(root, state_path).read_text())
            errors.extend(check_state(root, state, unit_id))
            if state.get("protocol") not in row["protocols"]:
                errors.append(f"{unit_id}: state protocol absent from catalog")
            if state.get("result") and state["result"] != row["reading_entry"]:
                errors.append(f"{unit_id}: reading entry differs from current result owner")
    prefix = manifest["archive_prefix"] + "/qwen3-vl-dense-enumeration/experiments/"
    imported = [e["archive"] for e in manifest["files"] if e["archive"].startswith(prefix)
                and e["archive"] != prefix + "index.md"
                and not e["archive"].startswith(prefix + "history/")]
    expected_ids = {p[len(prefix):].split("/", 1)[0] for p in imported}
    if not expected_ids.issubset(ids):
        errors.append(f"uncatalogued imported experiments: {sorted(expected_ids - ids)}")
    expected_protocols = {p for p in imported if Path(p).name == "unit.md"}
    if not expected_protocols.issubset(protocols):
        errors.append(f"uncatalogued imported protocols: {sorted(expected_protocols - protocols)}")
    actual_states = {str(p.relative_to(root)) for p in (root / program / "experiments").rglob("state.json")}
    if actual_states != states:
        errors.append(f"state/catalog mismatch: {sorted(actual_states ^ states)}")
    return errors


def links_in(text: str) -> list[str]:
    return LINK.findall(FENCED.sub("", text))


def check_live_links(root: Path, manifest: dict[str, Any], paths: list[Path]) -> tuple[list[str], int, int]:
    errors: list[str] = []
    local_count = external_count = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if re.search(r"@(E|A|H|ROOT|P|C|MANIFEST|GUIDE)(?:/|\))", text):
            errors.append(f"unexpanded link marker: {path.relative_to(root)}")
        for target in links_in(text):
            ref = resolve_reference(root, manifest, str(path.relative_to(root)), target)
            if ref["kind"] == "external":
                external_count += 1
            elif ref["kind"] != "local" or not ref["exists"]:
                errors.append(f"broken live link: {path.relative_to(root)} -> {target}")
            else:
                local_count += 1
    return errors, local_count, external_count


def historical_link_gaps(root: Path, manifest: dict[str, Any]) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    for entry in manifest["files"]:
        if not entry["archive"].endswith(".md"):
            continue
        for target in links_in(local_path(root, entry["archive"]).read_text(encoding="utf-8")):
            ref = resolve_reference(root, manifest, entry["archive"], target)
            if ref["kind"] != "external" and not ref["exists"]:
                gaps.append({"source": entry["source"], "target": target})
    return gaps


def run_check(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    errors = check_sources(root, manifest) + check_git_sources(root, manifest)
    archived = local_path(root, manifest["archive_prefix"])
    expected = {e["archive"] for e in manifest["files"] if e["action"] == "relocated_byte_exact"}
    actual = {str(p.relative_to(root)) for p in archived.rglob("*") if p.is_file()}
    if actual != expected:
        errors.append(f"frozen archive tree changed: {sorted(actual ^ expected)[:8]}")
    alias = manifest["compatibility_alias"]
    link = root / alias["path"]
    if not link.is_symlink() or os.readlink(link) != alias["target"]:
        errors.append("legacy alias is absent or changed")
    scanroot = link / "qwen3-vl-dense-enumeration"
    scan = [{"path": str(p.relative_to(root)), "sha256": digest(p.read_bytes())}
            for p in sorted(scanroot.rglob("*.json"))
            if "2026-09-12-parallel-owner-research" not in str(p)]
    if scan != manifest["legacy_json_scan"]:
        errors.append("legacy transfer-selector JSON scan changed")
    rows = [json.loads(line) for line in (root / PROGRAM / "experiments.jsonl").read_text().splitlines() if line.strip()]
    errors.extend(check_catalog(root, rows, manifest))
    live = [root / "research/index.md", root / "research/CONVENTIONS.md", root / CAPTURE / "README.md"]
    live.extend(sorted((root / PROGRAM).rglob("*.md")))
    link_errors, local_links, external = check_live_links(root, manifest, live)
    errors.extend(link_errors)
    for p in (root / PROGRAM).rglob("*"):
        if p.is_symlink() or p.suffix in {".py", ".pyc", ".sh", ".log"} or p.name == "__pycache__":
            errors.append(f"non-knowledge runtime file in live program: {p.relative_to(root)}")
    current = (root / PROGRAM / "current.md").read_text()
    current_targets = {resolve_reference(root, manifest, str(PROGRAM / "current.md"), t).get("path") for t in links_in(current)}
    for row in rows:
        if row["tracking"] == "current" and row["state"] not in current_targets:
            errors.append(f"current context omits state owner: {row['id']}")
    gaps = historical_link_gaps(root, manifest) if not errors else []
    return {"ok": not errors, "errors": errors, "preserved_source_files": len(manifest["files"]),
            "relocated_source_files": len(expected), "catalog_entries": len(rows),
            "catalogued_protocols": sum(len(r["protocols"]) for r in rows),
            "current_state_owners": sum(r["tracking"] == "current" for r in rows),
            "live_documents": len(live), "valid_live_local_links": local_links,
            "unverified_external_live_handles": external, "unchanged_legacy_json_paths": len(scan),
            "historical_unresolved_links": len(gaps), "historical_gap_examples": gaps[:6],
            "scope": "Local knowledge structure and original Git/source identity; no scientific replication or external-artifact verification."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("check", "resolve"), default="check")
    parser.add_argument("document", nargs="?")
    parser.add_argument("target", nargs="?")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        manifest = json.loads((root / CAPTURE / "manifest.json").read_text())
        if args.command == "resolve":
            if args.document is None or args.target is None:
                parser.error("resolve requires a source document and a link target")
            result = resolve_reference(root, manifest, args.document, args.target)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["kind"] == "external" or result.get("exists") else 1
        result = run_check(root, manifest)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
