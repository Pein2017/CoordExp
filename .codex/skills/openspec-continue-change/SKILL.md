---
name: openspec-continue-change
description: Use when the user wants to create the next artifact for an existing OpenSpec change.
---

# OpenSpec Continue Change

Requires `openspec` CLI.

Advance one OpenSpec artifact at a time. This skill is for artifact creation, not implementation.

## Flow

1. Select the change from explicit user input, recent conversation, or `openspec list --json`; ask if ambiguous.
2. Run:
   ```bash
   openspec status --change "<name>" --json
   ```
3. If all artifacts are complete, report that apply/verify/archive are available.
4. Pick the first ready artifact, then run:
   ```bash
   openspec instructions <artifact-id> --change "<name>" --json
   ```
5. Read listed dependencies, fill the template at `outputPath`, and stop after creating one artifact.

## Guardrails

- Follow CLI instructions but filter through CoordExp ownership rules: docs for stable guidance, progress for evidence, repo artifacts for executable truth.
- If the artifact would become ordinary task tracking rather than a compatibility contract, recommend a repo-local super-power plan instead.
- Do not mark implementation tasks complete here.
