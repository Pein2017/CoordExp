---
type: investigation
title: Docs vs MCP Authority Boundary
description: Establishes how CoordExp-Swift should use current docs, OKF research notes, and direct source inspection without making discovery compete with authority.
tags: [docs, authority, okf, coordexp-swift]
state: active
updated: 2026-07-07
---

# Docs vs MCP Authority Boundary

## Verdict

Keep `docs/` as the compact current-authority and routing surface. Use `rg`,
`git grep`, direct reads, AST/structured parsers, and focused tests or probes
for live code exploration and implementation impact analysis. Do not replace
the current docs spine with ad hoc source search, because source structure does
not encode project intent, accepted evidence gates, historical/current
boundaries, or operator policy by itself.

## Scope

This investigation is scoped to `/data/CoordExp/.worktrees/CoordExp-swift` and
to the question of how agents should balance:

- current-behavior docs and routing pages;
- stable OpenSpec contracts;
- OKF-style research notes;
- direct shell and structured source inspection as code exploration tools.

It does not rewrite `docs/`, change OpenSpec contracts, change source code, or
claim that the current docs tree is already optimally pruned.

## Current Recommendation

The near-term policy is preservation plus sharpening:

- preserve the Swift worktree docs spine: `docs/PROJECT_CONTEXT.md`,
  `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, `docs/COORDEXP_SWIFT.md`,
  `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, and relevant
  domain contracts;
- use `rg`/`git grep` first to narrow current-code maps;
- use direct symbol reads, AST/structured parsers, and focused executable probes
  when exact references, body reads, diagnostics, or edit precision are needed;
- demote stale dated plans, long execution records, and code mirrors only after
  a separate docs review, not as part of this OKF pilot.

## Interpretation

The right split is not "docs versus search." It is "authority versus discovery."
Docs answer what is current, what is canonical, what evidence is accepted, and
what must not be silently reinterpreted. MCPs answer where the live code is and
how it connects right now.

That distinction matters most in a worktree like CoordExp-Swift, where old
mainline routes, active rebuilt Swift routes, archived OpenSpec material, and
local research artifacts coexist. Code exploration can find all of those
surfaces, but it cannot decide which one is normative without a written
authority layer.
