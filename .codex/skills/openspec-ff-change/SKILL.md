---
name: openspec-ff-change
description: Use when the user explicitly asks to fast-forward OpenSpec artifacts for a stable CoordExp contract.
---

# OpenSpec Fast-Forward

Requires `openspec` CLI.

Create the complete apply-ready artifact set in one pass only when OpenSpec is explicitly in scope. Keep artifacts compact: proposal explains why, delta specs define stable contract changes, design records only decisions that affect compatibility, and tasks cover implementation plus verification.

## Flow

1. Require a clear change name or description. Derive kebab-case when needed.
2. Create or reuse the change:
   ```bash
   openspec new change "<name>"
   openspec status --change "<name>" --json
   ```
3. For each ready artifact, run `openspec instructions <artifact-id> --change "<name>" --json`, read dependency artifacts, and write only the requested output path.
4. Re-run status after each artifact and continue until all apply-required artifacts are done.
5. Summarize created files and the exact implementation entry point.

## CoordExp Rules

- Do not copy CLI `context` or `rules` blocks into artifacts.
- Do not encode future experiment gates as current requirements.
- Keep detailed implementation checklists in tasks or repo-local plans, not docs.
- Use `docs/PROJECT_CONTEXT.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, and relevant domain docs before inventing contract language.
