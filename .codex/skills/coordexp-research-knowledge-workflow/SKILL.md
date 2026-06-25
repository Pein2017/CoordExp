---
name: coordexp-research-knowledge-workflow
description: Use when CoordExp research knowledge must be collected, preserved, migrated, or synthesized across worktrees, progress notes, research notes, diagnostics, supervisor packets, or OKF-style research hubs without confusing historical evidence with current docs/spec authority.
---

# CoordExp Research Knowledge Workflow

Use this when the work product is durable research knowledge, not current-behavior docs, production code, or ordinary audit findings. Keep raw evidence, synthesized interpretation, and authoritative contracts separate.

## Role Boundary

Use for:

- cross-worktree Markdown union collection;
- migrating or reorganizing `progress/` evidence into `research/`;
- OKF-style research hubs, idea/investigation/mechanism notes, and continuation context;
- compressing many diagnostic notes into supervisor-facing packets;
- preserving provenance before cleanup, merge, or archive decisions.

Do not use this as the first skill for:

- current behavior docs or schema changes: use repo docs/OpenSpec routing;
- severity-ranked correctness audits: use `audit-review` or `model-innovation-risk-audit`;
- abnormal model behavior diagnosis: use `model-diagnosis`;
- isolated feature branches or cleanup lifecycle: use `worktree-feature-loop` and `git-hygiene`;
- infer/eval launch or artifact repair: use `coordexp-infer-eval-workflow`.

## Authority Rules

- `docs/` explains current behavior and operator guidance.
- `openspec/specs/` owns stable compatibility-sensitive contracts.
- `research/` is for ideas, mechanisms, investigations, interpretation, negative results, and continuation context.
- `progress/` is legacy evidence, diagnostics, benchmark history, and provenance.
- `docs/history/` may preserve raw intake snapshots, but those snapshots are non-normative.

When sources disagree, treat `research/` and `progress/` as evidence to explain or motivate; do not promote them to current-behavior authority without an explicit docs/spec update.

## Workflow

1. **Bound the source set.** Record the exact glob, worktree list, date window, branch set, artifact root, or user-named notes. Do not expand the scope silently.
2. **Inspect worktree state.** Use `git worktree list --porcelain` and focused status checks. Dirty worktrees are expected; ignore non-Markdown dirt unless it affects the requested evidence.
3. **Preserve raw provenance first.** For union collection, classify by content hash plus path, then snapshot new or divergent Markdown under a dated `docs/history/worktree-union/<date>/` bundle with a manifest.
4. **Separate phases.** Keep raw intake, migration design, synthesized research hubs, and supervisor packets in distinct files or commits.
5. **Synthesize intentionally.** Build reading paths under `research/` rather than mirrors of raw `progress/` files. Prefer `index.md`, `overview.md`, `draft.md`, `discussion.md`, `implementation.md`, `experiments/`, and `archive/` when they fit.
6. **Route discoveries.** Promote only stable current behavior into `docs/`; use OpenSpec only for stable compatibility-sensitive contracts.
7. **Verify the boundary.** Check source counts, manifest rows, tracked file set, local links, YAML/frontmatter where used, and ignored-file behavior before reporting.

## Union Collection Pattern

Use when the user wants all scattered Markdown gathered before cleanup or migration.

- Start from the exact checkout and worktree set.
- Include tracked branch-head Markdown plus dirty or untracked Markdown when the user asks for the union of knowledge.
- Classify at least:
  - `content_present_elsewhere`;
  - `new_content_new_path`;
  - `same_path_divergent_new_content`;
  - `same_path_identical`.
- Preserve raw snapshots as provenance, including trailing whitespace if it belongs to the original source.
- Write a manifest that records source worktree, branch, path, content hash, classification, and snapshot path.

Do not validate curated style against raw intake snapshots. Validate only that the manifest and snapshot boundary are accurate.

## OKF-Style Research Migration

Use a repo-native structure rather than depending on app-specific wiki syntax. Durable value comes from typed reading paths, clear authority boundaries, and linked evidence.

Default hierarchy:

```text
research/index.md
research/ideas/index.md
research/investigations/index.md
research/mechanisms/index.md
research/archive/index.md
research/ideas/<slug>/
```

Conventions:

- router `index.md` files may be plain;
- non-router notes may use light frontmatter such as `type: idea`, `type: investigation`, or `type: mechanism`;
- avoid premature `conclusion.md` unless the research direction is actually closed;
- update `docs/AGENT_INDEX.md` and `docs/catalog.yaml` only when `research/` discoverability or authority boundaries change.

If `/research/` is ignored, stage only the intended tracked set with explicit paths or `git add -f` after review. Do not sweep unrelated ignored local research artifacts into validation or commits.

## Supervisor Packet Mode

Use when the user needs many heavy diagnostics compressed for a supervisor or collaborator.

Good default packet:

```text
01_supervisor_brief.md
02_evidence_atlas.md
03_advice_and_next_steps.md
```

The brief should foreground decisions, strongest findings, demoted claims, and the launch or training gate. The atlas should group evidence by mechanism family with source-note and artifact handles. The advice memo should ask for concrete guidance on next tracks, thresholds, replications, or unblockers.

Place packet files outside the original nonrecursive source glob when the user gave one, so follow-up collection does not re-ingest its own synthesis.

## Commit And Cleanup Shape

When the work is large enough to commit, keep batches reviewable:

1. raw intake/provenance;
2. migration design/plan;
3. synthesized research pilot and router updates.

Do not infer push approval from local merge/cleanup approval. Use `git-hygiene` for staging, commits, sync, conflict resolution, or publication.

## Verification

Pick checks that match the product:

```bash
git worktree list --porcelain
find <source> -maxdepth 1 -type f -name '<glob>' | wc -l
wc -l <packet-or-research-files>
python - <<'PY'
import pathlib, yaml
for p in pathlib.Path("docs").glob("**/*.yaml"):
    yaml.safe_load(p.read_text())
PY
git check-ignore -v research/index.md || true
git ls-files research
git diff --check -- <touched-files>
```

For supervisor packets, also check headings, placeholder tokens, trailing whitespace, and balanced code fences.

## Output Contract

Report:

- source boundary and evidence date window;
- created or updated research/progress/history files;
- what stayed raw provenance versus synthesized interpretation;
- authority caveats and current-behavior docs/specs touched or intentionally untouched;
- verification commands and scope;
- unresolved evidence gaps and next continuation seeds.
