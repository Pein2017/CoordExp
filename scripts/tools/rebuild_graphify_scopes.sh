#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
SCOPES_FILE="${ROOT}/.codex/graphify/scopes.json"

if [[ ! -f "${SCOPES_FILE}" ]]; then
  echo "missing scopes file: ${SCOPES_FILE}" >&2
  exit 1
fi

python3 - "${ROOT}" "${SCOPES_FILE}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path

from graphify.analyze import god_nodes, suggest_questions, surprising_connections
from graphify.build import build
from graphify.cluster import cluster, score_all
from graphify.export import to_html, to_json
from graphify.extract import extract
from graphify.report import generate


def sanitize_token(value: str) -> str:
    return (
        value.replace("/", "__")
        .replace(".", "_dot_")
        .replace("-", "_dash_")
        .replace(" ", "_")
    )


def should_skip(rel: Path, excludes: list[str]) -> bool:
    rel_text = rel.as_posix()
    parts = set(rel.parts)
    for item in excludes:
        item = item.strip("/")
        if not item:
            continue
        if rel_text == item or rel_text.startswith(item + "/"):
            return True
        if item in parts:
            return True
    return False


def build_scope(root: Path, scope: dict, global_excludes: list[str]) -> dict:
    allowed = [Path(p) for p in scope["include_dirs"]]
    suffixes = set(scope["suffixes"])
    py_files: list[Path] = []
    other_files: list[Path] = []

    for base in allowed:
        abs_base = root / base
        if not abs_base.exists():
            continue
        for path in abs_base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if should_skip(rel, global_excludes):
                continue
            if path.suffix not in suffixes:
                continue
            if path.suffix == ".py":
                py_files.append(path)
            else:
                other_files.append(path)

    py_files = sorted(set(py_files))
    other_files = sorted(set(other_files))
    all_files = py_files + other_files

    ast = extract(py_files) if py_files else {"nodes": [], "edges": [], "hyperedges": [], "input_tokens": 0, "output_tokens": 0}

    scope_nodes = []
    file_nodes = []
    scaffold_edges = []

    scope_lookup: dict[str, str] = {}
    for base in allowed:
        abs_base = root / base
        if not abs_base.exists():
            continue
        rel_base = base.as_posix()
        scope_id = f"scope__{scope['name']}__{sanitize_token(rel_base)}"
        scope_lookup[rel_base] = scope_id
        scope_nodes.append(
            {
                "id": scope_id,
                "label": rel_base,
                "file_type": "document",
                "source_file": rel_base,
                "source_location": None,
                "source_url": None,
                "captured_at": None,
                "author": None,
                "contributor": None,
            }
        )

    file_node_ids: dict[str, str] = {}
    for path in all_files:
        rel = path.relative_to(root).as_posix()
        fid = f"file__{scope['name']}__{sanitize_token(rel)}"
        file_node_ids[rel] = fid
        file_type = "code" if path.suffix == ".py" else "document"
        file_nodes.append(
            {
                "id": fid,
                "label": rel,
                "file_type": file_type,
                "source_file": rel,
                "source_location": None,
                "source_url": None,
                "captured_at": None,
                "author": None,
                "contributor": None,
            }
        )
        top = rel.split("/", 1)[0]
        scope_id = scope_lookup.get(top)
        if scope_id:
            scaffold_edges.append(
                {
                    "source": scope_id,
                    "target": fid,
                    "relation": "references",
                    "confidence": "EXTRACTED",
                    "confidence_score": 1.0,
                    "source_file": rel,
                    "source_location": None,
                    "weight": 1.0,
                }
            )

    for node in ast.get("nodes", []):
        src = node.get("source_file")
        if src and src in file_node_ids:
            scaffold_edges.append(
                {
                    "source": file_node_ids[src],
                    "target": node["id"],
                    "relation": "references",
                    "confidence": "EXTRACTED",
                    "confidence_score": 1.0,
                    "source_file": src,
                    "source_location": node.get("source_location"),
                    "weight": 1.0,
                }
            )

    scaffold = {
        "nodes": scope_nodes + file_nodes,
        "edges": scaffold_edges,
        "hyperedges": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }

    graph = build([ast, scaffold])
    communities = cluster(graph)
    cohesion_scores = score_all(graph, communities)
    labels = {nid: data.get("label", nid) for nid, data in graph.nodes(data=True)}
    community_labels = {}
    for cid, members in communities.items():
        ranked = sorted(members, key=lambda n: (graph.degree(n), labels.get(n, n)), reverse=True)
        community_labels[cid] = " / ".join(labels.get(n, n) for n in ranked[:3]) if ranked else f"community_{cid}"

    report_text = generate(
        graph,
        communities,
        cohesion_scores,
        community_labels,
        god_nodes(graph),
        surprising_connections(graph, communities),
        {
            "total_files": len(all_files),
            "total_words": 0,
            "files": {
                "code": [p.relative_to(root).as_posix() for p in py_files],
                "document": [p.relative_to(root).as_posix() for p in other_files],
                "paper": [],
                "image": [],
            },
            "warning": scope["description"],
            "needs_graph": True,
            "graphifyignore_patterns": [],
            "skipped_sensitive": [],
        },
        {"input": 0, "output": 0},
        f"{root.name}:{scope['name']}",
        suggested_questions=suggest_questions(graph, communities, community_labels),
    )

    out_dir = root / "graphify-out" / "scopes" / scope["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    to_json(graph, communities, str(out_dir / "graph.json"))
    to_html(graph, communities, str(out_dir / "graph.html"), community_labels)
    (out_dir / "GRAPH_REPORT.md").write_text(report_text, encoding="utf-8")
    (out_dir / "SCOPE.json").write_text(
        json.dumps(
            {
                "name": scope["name"],
                "description": scope["description"],
                "include_dirs": scope["include_dirs"],
                "suffixes": scope["suffixes"],
                "global_excludes": global_excludes,
                "py_files": len(py_files),
                "document_files": len(other_files),
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
                "communities": len(communities),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "name": scope["name"],
        "description": scope["description"],
        "path": out_dir.relative_to(root).as_posix(),
        "py_files": len(py_files),
        "document_files": len(other_files),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "communities": len(communities),
    }


root = Path(__import__("sys").argv[1]).resolve()
scopes_file = Path(__import__("sys").argv[2]).resolve()
config = json.loads(scopes_file.read_text(encoding="utf-8"))
global_excludes = config.get("global_excludes", [])
results = []
for scope in config.get("scopes", []):
    results.append(build_scope(root, scope, global_excludes))

index_path = root / "graphify-out" / "scopes" / "INDEX.json"
index_path.parent.mkdir(parents=True, exist_ok=True)
index_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

for item in results:
    print(
        f"{item['name']}: py={item['py_files']} doc={item['document_files']} "
        f"nodes={item['graph_nodes']} edges={item['graph_edges']} communities={item['communities']}"
    )
PY
