---
type: investigation
title: Docs vs MCP Authority Maintenance Guidance
description: Practical maintenance rules for keeping CoordExp-Swift docs useful without duplicating MCP-powered code exploration.
tags: [docs, mcp, maintenance, authority, codegraph, serena]
updated: 2026-07-07
---

# Docs vs MCP Authority Maintenance Guidance

## Principle

Maintain docs as authority and routing, not as a hand-written source index.
Maintain MCPs as live implementation discovery, not as a substitute for project
intent or accepted evidence.

## Docs Own

Docs should own:

- current behavior and operator guidance;
- canonical entrypoints and route choices;
- validation gates and accepted evidence handles;
- artifact names, metric semantics, and provenance expectations;
- historical versus current boundaries;
- future-work boundaries and explicit non-claims.

Docs should be pointer-first. A good current doc names the right entrypoint,
contract, artifact, or check, then lets CodeGraph, Serena, tests, and source
reads carry the implementation detail.

## MCPs Own

CodeGraph and Serena should own:

- live code navigation;
- symbol lookup and grouped source context;
- callers, callees, impact, and dependency exploration;
- exact Python symbol references and diagnostics after a target file is known;
- implementation verification support before edits or reviews.

Before using CodeGraph for a worktree-specific claim, check that the index
belongs to the exact worktree and is fresh enough for the files under review.
Use Serena for narrowed Python precision, not broad repo governance.

## What To Prune Later

A later docs review can demote or move material when it is:

- a dated implementation plan rather than current behavior;
- a long execution record rather than a route or contract;
- a function/class mirror that MCPs can regenerate more accurately;
- historical evidence better suited to `progress/`, `docs/history/`, or
  `research/`;
- duplicated guidance that belongs in one canonical page.

Do not prune solely because a topic is searchable. Prune when a doc no longer
answers an authority, routing, evidence, or maintenance question.

## Update Rules

Update current docs when behavior, schema, artifact names, metric semantics,
entrypoints, accepted validation gates, or recommended workflows change.

Update `research/` when the product is interpretation, investigation,
mechanism explanation, negative result, or continuation context.

Update `openspec/specs/` only for stable compatibility-sensitive contracts.

Leave raw historical evidence in `progress/` or `docs/history/` unless it has
been synthesized into an OKF reading path.

## Agent Workflow

For CoordExp-Swift work:

1. Open `docs/PROJECT_CONTEXT.md`, `docs/AGENT_INDEX.md`, and
   `docs/COORDEXP_SWIFT.md` to establish authority.
2. Use `docs/catalog.yaml` and `docs/IMPLEMENTATION_MAP.md` to narrow the
   relevant route.
3. Check CodeGraph freshness before broad implementation mapping.
4. Use Serena or exact source reads for narrowed Python semantics.
5. Verify claims with tests, artifacts, manifests, metrics, or stable specs.

This keeps docs small and durable while letting MCPs do the heavy lifting for
live code exploration.
