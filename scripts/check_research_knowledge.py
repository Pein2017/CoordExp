"""Validate the current research catalog, topic links and exact Git recovery.

Read-only. Historical recovery never establishes continuation authority. The
catalog is the only inventory; this checker creates no state or archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

CATALOG = "research/experiments/catalog.jsonl"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def local_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or any(c in name for c in "\0\r\n\\"):
        raise ValueError("invalid repository path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError("path must be normalized and repository-relative")
    result = root.joinpath(*path.parts)
    if result.is_symlink() or not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("path escapes repository")
    return result


def _git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    result = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args],
                            input=stdin, capture_output=True, timeout=60)
    if result.returncode:
        raise ValueError("historical Git source is not available")
    return result.stdout


def check_catalog(root: Path, rows: list[dict]) -> list[str]:
    errors, seen, recoveries = [], set(), {}
    current_states = set()
    for row in rows:
        if not isinstance(row, dict):
            errors.append("catalog row must be an object"); continue
        label = row.get("id", "<missing>")
        try:
            if not isinstance(label, str) or not ID.fullmatch(label) or label in seen:
                raise ValueError("invalid or duplicate catalog ID")
            seen.add(label)
            if row.get("schema_version") != 2 or not isinstance(row.get("title"), str) or not row["title"]:
                raise ValueError("unsupported catalog schema/title")
            topics = row.get("topics")
            if not isinstance(topics, list) or not topics or len(set(topics)) != len(topics):
                raise ValueError("topics must be a nonempty unique list")
            for topic in topics:
                if not isinstance(topic, str) or not ID.fullmatch(topic):
                    raise ValueError("invalid topic ID")
                if not local_path(root, f"research/questions/{topic}.md").is_file():
                    raise ValueError("missing authoritative question")
            if row.get("tracking") == "distilled":
                if row.get("continuation") != "unsupported_historical":
                    raise ValueError("historical continuation must be unsupported")
                if not row.get("evidence") or not row.get("scientific_status") or not row.get("boundary") or not row.get("summary"):
                    raise ValueError("historical evidence/status/boundary/summary missing")
                source = row["recovery"]
                commit = source["commit"]
                if not isinstance(commit, str) or not HEX40.fullmatch(commit):
                    raise ValueError("full historical Git commit required")
                base = source["record_root"]
                local_path(root, base)
                if not base.startswith("research/experiments/") and not (row.get("kind") == "historical_synthesis" and base == "progress/diagnostics"):
                    raise ValueError("recovery root is not an original unit")
                paths = [source["reading_entry"], *source["protocols"], *source["results"]]
                for path in paths:
                    local_path(root, path)
                    if not path.startswith("research/") and not (row.get("kind") == "historical_synthesis" and path.startswith("progress/diagnostics/")):
                        raise ValueError("historical record is not research content")
                if not HEX64.fullmatch(source["reading_sha256"]):
                    raise ValueError("original reading hash required")
                if row.get("locator_status") != "recorded_not_revalidated" or not isinstance(row.get("external_locators"), list):
                    raise ValueError("external locator scope must be explicit")
                if any(not isinstance(p, str) or not p.startswith("/data/") or "\n" in p for p in row["external_locators"]):
                    raise ValueError("malformed external locator")
                recoveries.setdefault(commit, []).append((label, source, paths))
            elif row.get("tracking") == "current":
                record = local_path(root, row["record_root"])
                reading = local_path(root, row["reading_entry"])
                state_path = local_path(root, row["state"])
                if not record.is_dir() or not reading.is_file() or not state_path.is_file():
                    raise ValueError("current unit requires live record, reading and state")
                if not row["record_root"].startswith("research/experiments/") or record.name != label:
                    raise ValueError("current unit root must match its ID")
                if not reading.resolve().is_relative_to(record.resolve()) or not state_path.resolve().is_relative_to(record.resolve()):
                    raise ValueError("current state/reading must belong to the unit")
                if row["state"] in current_states:
                    raise ValueError("current state has multiple owners")
                current_states.add(row["state"])
                state = json.loads(state_path.read_text())
                if state.get("unit_id") != label or state.get("lifecycle") not in {"planned", "ready", "running", "blocked", "paused", "closed", "superseded"}:
                    raise ValueError("current state identity/lifecycle differs")
                if state.get("evidence") not in {"none", "partial", "unreviewed", "accepted", "invalid"}:
                    raise ValueError("current evidence status invalid")
                if not state.get("boundary") or not isinstance(state.get("not_authorized"), list):
                    raise ValueError("current state must declare its boundary and exclusions")
                for field in ("protocol", "state_source"):
                    if not local_path(root, state[field]).is_file():
                        raise ValueError("current state source/protocol missing")
                if state.get("result"):
                    if not local_path(root, state["result"]).is_file():
                        raise ValueError("current result is missing")
                elif state["evidence"] == "accepted":
                    raise ValueError("accepted current evidence requires a result")
            else:
                raise ValueError("catalog entry must be current or distilled")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(f"{label}: {exc}")
    for commit, records in recoveries.items():
        try:
            entries = {}
            for raw in _git(root, "ls-tree", "-r", "-z", commit).split(b"\0"):
                if raw:
                    meta, name = raw.split(b"\t", 1)
                    mode, kind, blob = meta.decode().split()
                    entries[name.decode()] = (mode, kind, blob)
            readings = []
            for label, source, paths in records:
                if any(path not in entries or entries[path][0] not in {"100644", "100755"} or entries[path][1] != "blob" for path in paths):
                    raise ValueError(f"{label}: missing regular historical record")
                readings.append(entries[source["reading_entry"]][2])
            output = _git(root, "cat-file", "--batch", stdin=("\n".join(readings) + "\n").encode())
            offset = 0
            for label, source, _ in records:
                end = output.index(b"\n", offset)
                header = output[offset:end].decode().split()
                if header[1] != "blob":
                    raise ValueError("recovery object is not a blob")
                size = int(header[2]); offset = end + 1
                data = output[offset:offset + size]; offset += size + 1
                if hashlib.sha256(data).hexdigest() != source["reading_sha256"]:
                    raise ValueError(f"{label}: original reading bytes changed")
        except (ValueError, OSError, IndexError) as exc:
            errors.append(f"{commit}: {exc}")
    live_states = {str(p.relative_to(root)) for p in (root / "research/experiments").rglob("state.json")}
    if current_states != live_states:
        errors.append("orphan or missing live state owner")
    return errors


def visible_text(text: str) -> str:
    text = re.sub(r"(?s)<!--.*?-->", "", text)
    return re.sub(r"(?ms)^\s*```[^\n]*\n.*?^\s*```[^\n]*$", "", text)


def check(root: Path) -> dict:
    root = root.resolve()
    rows = [json.loads(line) for line in local_path(root, CATALOG).read_text().splitlines() if line.strip()]
    errors = check_catalog(root, rows)
    ids = {row["id"] for row in rows}
    docs = [root / name for name in ("research/index.md", "research/story.md", "research/CONVENTIONS.md", "research/assets.md", "research/glossary.md")]
    docs += sorted((root / "research/questions").glob("*.md"))
    references, external = 0, 0
    for document in docs:
        if not document.is_file():
            errors.append(f"missing authoritative document: {document.name}"); continue
        text = visible_text(document.read_text())
        for ref in re.findall(r"catalog:([A-Za-z0-9_.-]+)", text):
            ref = ref.rstrip(".")
            if ref not in ids:
                errors.append(f"{document.name}: missing catalog ID {ref}")
            references += 1
        for target in re.findall(r"!?\[[^\]\n]+\]\(([^)\n]+)\)", text):
            target = urlsplit(target.strip().strip("<>"))
            if target.scheme or target.netloc or target.path.startswith("/"):
                external += 1; continue
            path = (document.parent / unquote(target.path)).resolve() if target.path else document
            if not path.is_relative_to(root) or not path.exists():
                errors.append(f"{document.name}: missing/escaping current link {target.path}")
    index = visible_text((root / "research/index.md").read_text())
    inline_free = re.sub(r"`+[^`\n]*`+", "", index)
    navigation = [urlsplit(p.strip().strip("<>")) for p in re.findall(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)", inline_free)]
    catalog_link = any(not p.scheme and not p.netloc and not p.path.startswith("/") and
                       (root / "research" / unquote(p.path)).resolve() == (root / CATALOG).resolve()
                       for p in navigation)
    if not catalog_link:
        errors.append("index must link the catalog")
    for question in (root / "research/questions").glob("*.md"):
        if f"questions/{question.name}" not in index:
            errors.append(f"index omits question {question.name}")
    return {"ok": not errors, "errors": errors, "catalog_entries": len(rows),
            "distilled": sum(row["tracking"] == "distilled" for row in rows),
            "current": sum(row["tracking"] == "current" for row in rows),
            "claim_references": references, "external_links_not_revalidated": external,
            "scope": "current knowledge and exact historical Git recovery; never execution qualification"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check",))
    parser.parse_args()
    try:
        result = check(Path(__file__).resolve().parents[1])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"ok": False, "errors": [str(exc)]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(not result["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
