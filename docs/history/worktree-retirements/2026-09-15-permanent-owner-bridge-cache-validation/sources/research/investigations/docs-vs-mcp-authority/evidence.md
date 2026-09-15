---
type: investigation
title: Docs vs MCP Authority Evidence
description: Live evidence snapshot for the CoordExp-Swift docs and MCP boundary decision.
tags: [docs, codegraph, serena, evidence, coordexp-swift]
updated: 2026-07-07
---

# Docs vs MCP Authority Evidence

Evidence scope: live inspection of `/data/CoordExp/.worktrees/CoordExp-swift`
on 2026-07-07. This is synthesized interpretation, not raw provenance intake.

## Docs Surface

The worktree still has a substantial tracked docs tree:

| field | value |
| --- | ---: |
| tracked docs files | 254 |
| docs file count | 254 |
| docs directory size | 7.8M |

Key current-authority and routing files:

- `docs/PROJECT_CONTEXT.md`: defines authority boundaries and read order.
- `docs/AGENT_INDEX.md`: agent-facing route map.
- `docs/catalog.yaml`: machine-readable curated inventory.
- `docs/COORDEXP_SWIFT.md`: Swift worktree authority page.
- `docs/SYSTEM_OVERVIEW.md`: system-level orientation.
- `docs/IMPLEMENTATION_MAP.md`: task-to-file routing.
- `docs/ARTIFACTS.md`: artifact and provenance handles.

## Swift Worktree Authority Role

`docs/COORDEXP_SWIFT.md` is not redundant with code search. It records:

- the current V1 verdict for the rebuilt Swift backbone;
- active source topology and config routes;
- the accepted fixed val200 validation gate;
- accepted metric handles and values;
- stabilized contracts such as source-order geometry, pack-cache identity, and
  neutral inference defaults;
- future-work boundaries that should not be inferred from code structure alone.

These are governance and interpretation facts. They are not reliably derivable
from symbol graphs.

## MCP State

The worktree has a local CodeGraph index:

| field | value |
| --- | ---: |
| indexed files | 1,309 |
| nodes | 23,068 |
| edges | 53,550 |
| database size | 65.84 MB |
| language files | 985 Python, 324 YAML |

The same status report showed pending index drift:

| pending change kind | count |
| --- | ---: |
| added files | 4 |
| modified files | 16 |

CodeGraph also reported that the index was built by an earlier engine version
and recommended `codegraph sync` or a full rebuild before relying on the graph
for fresh implementation claims.

This is expected and not a CodeGraph failure. It means graph exploration is a
live-code discovery tool whose freshness must be checked in the exact worktree.
It should not replace docs as the stable authority layer.

## Boundary Evidence

This worktree contains both current Swift routes and legacy/mainline routes.
Without a written authority layer, a graph or symbol search can easily surface
historical paths alongside current ones. The docs spine prevents that ambiguity
by identifying which routes are current, which are historical, and which
validation evidence is accepted.

The evidence supports a two-layer workflow:

- read docs first to establish authority and scope;
- use MCPs next to inspect and verify the current implementation.
