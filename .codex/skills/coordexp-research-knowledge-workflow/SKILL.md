---
name: coordexp-research-knowledge-workflow
description: Use to collect, preserve, migrate, or synthesize CoordExp research knowledge across worktrees, legacy notes, diagnostics, supervisor packets, or research hubs while keeping historical evidence separate from current docs/spec authority.
---

# CoordExp Research Knowledge Workflow

Use this when the deliverable is durable research knowledge rather than current
behavior docs, production code, or an ordinary audit.

## Authority

- `docs/`: current behavior and operator guidance.
- `openspec/specs/`: stable compatibility-sensitive contracts.
- `research/`: active ideas, investigations, mechanisms, interpretation,
  negative results, and continuation context.
- `progress/`: deprecated legacy evidence; read only for reconstruction, migrate
  useful material, and do not create new records.
- `docs/history/`: raw non-normative provenance snapshots.

Use `git-hygiene` for worktree isolation, staging, commits, sync, cleanup, or
publication. Use `audit-review` before promoting a synthesized claim into
current docs or stable specs.

## Workflow

1. Bound exact worktrees, branches, globs, dates, notes, or artifact roots.
2. Inspect each target checkout independently; do not mix checkout facts.
3. Preserve raw provenance before synthesis. For a Markdown union, classify by
   content hash and path, then snapshot new/divergent material under
   `docs/history/worktree-union/<date>/` with a manifest.
4. Keep raw intake, migration decisions, synthesis, and supervisor packets
   distinct. Build reading paths under `research/`, not mirrors of `progress/`.
5. Route only verified current behavior to `docs/`; use OpenSpec only when a
   stable compatibility contract changes.
6. Verify source counts, manifest rows, links, frontmatter/YAML, tracked/ignored
   boundaries, and the scoped diff.

For union manifests record source worktree, branch, path, content hash,
classification, and snapshot path. Useful classifications are
`content_present_elsewhere`, `new_content_new_path`,
`same_path_divergent_new_content`, and `same_path_identical`. Preserve original
snapshot bytes; validate curation rules only on synthesized files.

## Research shape

A repo-native hub usually needs only the applicable paths:

```text
research/index.md
research/ideas/index.md
research/investigations/index.md
research/mechanisms/index.md
research/archive/index.md
research/ideas/<slug>/
```

Use light typed frontmatter on non-router notes when helpful. Avoid a
`conclusion.md` until the direction is actually closed. Update
`docs/AGENT_INDEX.md` or `docs/catalog.yaml` only when discoverability or
authority boundaries change.

For a supervisor packet, default to a short brief, an evidence atlas, and an
advice/next-steps memo. Foreground decisions, strongest evidence, demoted
claims, gates, exact artifact handles, and concrete questions. Place synthesis
outside a user-supplied nonrecursive source glob so it cannot re-ingest itself.

## Verification and report

Use only checks relevant to the product, for example:

```bash
git worktree list --porcelain
git check-ignore -v research/index.md || true
git ls-files research
git diff --check -- <touched-files>
```

Report the source/evidence boundary, raw versus synthesized outputs, migrated or
untouched legacy sources, authority caveats, checks run, unresolved evidence
gaps, and continuation seeds. Never infer push approval from local synthesis or
cleanup approval.
