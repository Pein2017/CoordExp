#!/usr/bin/env python3
"""Validate the lightweight decision layer of the research graph."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


REQUIRED_HEADINGS = (
    "## Decision",
    "## Evidence",
    "## Belief Update",
    "## Next Discriminator",
)
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def split_frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("missing YAML frontmatter")
    try:
        raw, body = text[4:].split("\n---\n", 1)
    except ValueError as exc:
        raise ValueError("unterminated YAML frontmatter") from exc
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("frontmatter must be a mapping")
    return data, body


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    decision_dir = repo / "research" / "decisions"
    paths = sorted(p for p in decision_dir.glob("*.md") if p.name != "index.md")
    errors: list[str] = []
    nodes: dict[str, tuple[Path, dict]] = {}

    for path in paths:
        rel = path.relative_to(repo)
        try:
            data, body = split_frontmatter(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"{rel}: {exc}")
            continue
        decision_id = data.get("id")
        if not isinstance(decision_id, str) or not decision_id:
            errors.append(f"{rel}: missing non-empty id")
        elif decision_id in nodes:
            errors.append(f"{rel}: duplicate id {decision_id}")
        else:
            nodes[decision_id] = (path, data)
        if data.get("type") != "decision":
            errors.append(f"{rel}: type must be decision")
        if data.get("status") not in {"active", "revisit", "retired"}:
            errors.append(f"{rel}: invalid status {data.get('status')!r}")
        for heading in REQUIRED_HEADINGS:
            if heading not in body:
                errors.append(f"{rel}: missing heading {heading}")
        evidence = data.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{rel}: evidence must be a non-empty list")
        else:
            for item in evidence:
                if not isinstance(item, str) or not (repo / item).is_file():
                    errors.append(f"{rel}: unresolved evidence path {item!r}")
        for target in LINK_RE.findall(body):
            if target.startswith(("http://", "https://", "#", "/")):
                continue
            clean = target.split("#", 1)[0]
            if clean and not (path.parent / clean).exists():
                errors.append(f"{rel}: unresolved Markdown link {target}")

    edge_count = 0
    for decision_id, (path, data) in nodes.items():
        relations = data.get("relations")
        rel = path.relative_to(repo)
        if not isinstance(relations, dict):
            errors.append(f"{rel}: relations must be a mapping")
            continue
        for kind in ("supports", "narrows", "supersedes"):
            targets = relations.get(kind)
            if not isinstance(targets, list):
                errors.append(f"{rel}: relations.{kind} must be a list")
                continue
            for target in targets:
                edge_count += 1
                if target not in nodes:
                    errors.append(
                        f"{rel}: relations.{kind} references unknown id {target!r}"
                    )
                if target == decision_id:
                    errors.append(f"{rel}: self-relation through {kind}")

    if errors:
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        print(
            f"research_graph_invalid decisions={len(nodes)} edges={edge_count} "
            f"errors={len(errors)}",
            file=sys.stderr,
        )
        return 1
    print(f"research_graph_ok decisions={len(nodes)} edges={edge_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
