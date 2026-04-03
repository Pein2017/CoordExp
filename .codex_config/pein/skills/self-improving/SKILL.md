---
name: self-improving
description: Learn conservative reusable preferences, corrections, and workflow lessons for Codex CLI using a local repo-scoped memory store. Use when the user explicitly references this skill, asks to remember a reusable preference or correction, wants to inspect what has been learned, or wants repeated mistakes captured and reused safely.
---

# Self-Improving For Codex

This is a Codex-native adaptation of the OpenClaw `self-improving` skill.

Keep the skill itself portable under `.codex_config/pein/skills/self-improving/`.
Store mutable memory outside the skill directory in a workspace-local folder.

Default local memory root:

```text
.self-improving/
```

If a workspace already has a different local memory root, use that consistently instead.
Treat the chosen memory root as the only writable memory surface for this skill.
That local memory root may be git-tracked if the workspace intentionally wants to share memory across machines.

## Goals

- retain explicit user preferences that should survive across tasks in this workspace
- capture reusable corrections after mistakes or rework
- keep scoped lessons in the smallest valid namespace
- improve execution quality without overriding repo governance

## Priority And Scope

Follow normal instruction precedence first:

1. system and developer instructions
2. workspace `AGENTS.md`
3. canonical repo docs such as `docs/PROJECT_CONTEXT.md`
4. this skill's memory files

Memory from this skill is advisory. It never overrides higher-priority instructions.
The skill definition should stay portable; only the local memory root should vary by workspace.

## When To Activate

Use this skill when one of these is true:

- the user explicitly names `self-improving`
- the user asks to remember a reusable preference, correction, or workflow
- the user asks what has been learned or wants memory stats
- a repeated mistake or repeated successful workflow should be captured for future Codex sessions in this workspace

Do not activate just because a task exists. Keep this skill conservative.

## Storage Layout

By default, all memory lives in `.self-improving/`:

```text
.self-improving/
├── memory.md        # global hot rules and confirmed preferences
├── corrections.md   # explicit corrections and repeat tracking
├── reflections.md   # post-task lessons that may later graduate
├── index.md         # lightweight index and counts
├── projects/        # project-specific overrides
├── domains/         # domain-specific rules such as code or writing
└── archive/         # inactive or superseded items
```

## Loading Rules

Before using memory:

1. Read `memory.md`.
2. Read only the smallest matching scope:
   - `domains/<domain>.md` when the lesson is domain-specific
   - `projects/<project>.md` when the lesson is workspace- or project-specific
3. Do not bulk-load unrelated files.

## Learning Rules

Write only when the lesson is reusable.

- Explicit correction:
  append a concise entry to `corrections.md` immediately.
- Explicit lasting preference or rule:
  add it to `memory.md` if global, `domains/*.md` if domain-scoped, or `projects/*.md` if project-scoped.
- Self-reflection after meaningful work:
  add a short candidate lesson to `reflections.md` when it is likely to help later.

Do not learn from silence.
Do not infer durable preferences from one-off requests.
Do not store secrets, third-party personal data, or speculative psychological profiles.

## Promotion Rules

- one occurrence: keep as a correction or reflection candidate
- repeated pattern: keep tracking in `corrections.md`
- only promote to `memory.md`, `domains/`, or `projects/` when the user made the preference explicit or repetition is strong and non-ambiguous
- if the scope is unclear, prefer the narrower namespace

## Transparency

When a learned rule materially affects behavior, say so briefly and cite the source file when useful.
Example: `Using the repo-local self-improving rule from projects/coordexp.md`.

## Shared Memory Guardrail

If the local memory root is committed to git, treat it as shared project state rather than private scratch space.
Keep entries professional, concise, and safe for collaborators or other machines that may sync the branch.

## Available References

- safety boundaries: `references/boundaries.md`
- memory templates: `references/memory-template.md`
- operations and maintenance: `references/operations.md`

## Codex Tailoring

This port intentionally removes OpenClaw-only concepts such as:

- `clawhub install ...`
- `SOUL.md`
- `HEARTBEAT.md`
- global memory under `~/self-improving/`

Use the portable-skill plus local-memory split above instead.
