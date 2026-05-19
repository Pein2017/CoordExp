---
name: openspec-sync-specs
description: Use when applying an OpenSpec delta spec into stable main specs without archiving the change.
---

# OpenSpec Sync Specs

Requires `openspec` CLI.

Merge delta specs into `openspec/specs/` only for stable, compatibility-sensitive contracts. This is an agent merge, not a mechanical copy.

## Flow

1. Select the change; ask if ambiguous.
2. Find delta specs under `openspec/changes/<name>/specs/*/spec.md`.
3. For each capability, read the delta and current main spec before editing.
4. Apply:
   - `ADDED`: add or update the requirement if it already exists;
   - `MODIFIED`: patch only the named requirement/scenario content;
   - `REMOVED`: remove the named requirement block;
   - `RENAMED`: rename the requirement and preserve scenarios.
5. Summarize changed capabilities and requirements.

## Guardrails

- Preserve main-spec content not mentioned by the delta.
- Keep the operation idempotent.
- Do not sync experiment notes, implementation checklists, or unvalidated benchmark claims into stable specs.
