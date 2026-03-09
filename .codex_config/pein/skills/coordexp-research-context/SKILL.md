---
name: coordexp-research-context
description: Build broad project context by combining the canonical docs layer with the structured progress catalog.
---

# CoordExp Research Context

Use this skill when the user needs broad background, design history, or empirical context before implementation or audit work.

## Primary Entry Points

1. `docs/AGENT_INDEX.md`
2. `docs/catalog.yaml`
3. `progress/index.yaml`
4. `docs/PROJECT_CONTEXT.md`

## Read Order

Start here:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/SYSTEM_OVERVIEW.md`
3. `docs/IMPLEMENTATION_MAP.md`
4. the relevant domain router under `docs/`
5. relevant `openspec/specs/*.md`
6. `progress/index.yaml`
7. the specific historical notes you need

## Precedence

When two sources disagree, use:

1. `openspec/specs/`
2. `docs/`
3. `openspec/changes/<active-change>/`
4. `progress/`

## What To Produce

Build a short context pack with:

- current authoritative docs/specs
- relevant implementation files or configs
- the exact historical notes that matter
- a small set of grep seeds for narrowing

## Progress Usage

Use `progress/` for:

- historical design directions
- diagnostics and audits
- benchmark evidence
- pretraining background

Do not use it as the primary source for current behavior.
