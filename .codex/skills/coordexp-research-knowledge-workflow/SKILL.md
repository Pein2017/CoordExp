---
name: coordexp-research-knowledge-workflow
description: Use when CoordExp research knowledge must be collected, preserved, migrated, or synthesized across worktrees, legacy progress notes, research notes, diagnostics, supervisor packets, or Open Knowledge Format (OKF)-style research hubs without confusing historical evidence with current docs/spec authority.
---

# CoordExp Research Knowledge Workflow

Use this when the work product is durable research knowledge, not current-behavior docs, production code, or ordinary audit findings. Keep raw evidence, synthesized interpretation, and authoritative contracts separate.

## Role Boundary

Use for:

- cross-worktree Markdown union collection;
- migrating or reorganizing legacy `progress/` evidence into `research/`;
- Open Knowledge Format (`OKF`)-style research hubs,
  idea/investigation/mechanism notes, and continuation context;
- compressing many diagnostic notes into supervisor-facing packets;
- preserving provenance before cleanup, merge, or archive decisions.

Do not use this as the first skill for:

- current behavior docs or schema changes: use repo docs/OpenSpec routing;
- severity-ranked correctness audits: use `audit-review` or `model-innovation-risk-audit`;
- abnormal model behavior diagnosis: use `model-diagnosis`;
- isolated feature branches or cleanup lifecycle: use `worktree-feature-loop` and `git-hygiene`;
- infer/eval launch or artifact repair: use `coordexp-infer-eval-workflow`.

For large note clustering, cross-worktree collection, supervisor packets, or
Open Knowledge Format (`OKF`)-style hub drafts, delegate a bounded read-only
research/provenance lane with exact sources and an output schema. Before
promoting a synthesized claim into current docs or stable specs, use one
independent contract-focused lane. Select models and reasoning effort from
`.codex/MODEL_ROUTING.md`; do not depend on a permanent custom-agent profile.

## Authority Rules

- `docs/` explains current behavior and operator guidance.
- `openspec/specs/` owns stable compatibility-sensitive contracts.
- `research/` is for ideas, mechanisms, investigations, interpretation, negative results, and continuation context.
- `progress/` is deprecated legacy evidence, diagnostics, benchmark history, and
  provenance. Do not create new `progress/` records.
- `docs/history/` may preserve raw intake snapshots, but those snapshots are non-normative.

When sources disagree, treat `research/` as active interpretation and
`progress/` as legacy provenance; do not promote either to current-behavior
authority without an explicit docs/spec update.

## Workflow

1. **Bound the source set.** Record the exact glob, worktree list, date window, branch set, artifact root, or user-named notes. Do not expand the scope silently.
2. **Inspect worktree state.** Use `git worktree list --porcelain` and focused status checks. Dirty worktrees are expected; ignore non-Markdown dirt unless it affects the requested evidence.
3. **Preserve raw provenance first.** For union collection, classify by content hash plus path, then snapshot new or divergent Markdown under a dated `docs/history/worktree-union/<date>/` bundle with a manifest.
4. **Separate phases.** Keep raw intake, migration design, synthesized research hubs, and supervisor packets in distinct files or commits.
5. **Synthesize intentionally.** Build reading paths under `research/` rather than mirrors of raw legacy `progress/` files. Prefer `index.md`, `overview.md`, `draft.md`, `discussion.md`, `implementation.md`, `experiments/`, and `archive/` when they fit.
6. **Close units explicitly.** Separate planned protocol, executed evidence,
   interpretation, decision update, mechanism promotion, and implementation
   authorization. Read
   [Research Graph and Unit Contract](references/research-graph-contract.md)
   when creating or closing a unit.
7. **Close semantic deltas before freeze.** Apply the originating-intent and
   semantic-delta gate in the research graph contract before a unit becomes
   `ready` or freezes a cohort, predicate, estimand, control, claim, or stop
   rule. Treat handoffs and reviewer suggestions as derived routing context,
   not user-owned requirements. Then record the decision-owning outcome,
   intervention or proxy, final evaluation surface, transfer claim,
   preservation risks, and nested signal supply. Read
   [Research Alignment Examples](references/research-alignment-examples.md)
   when positive and negative patterns would clarify the boundary.
8. **Match the evidence tier.** Keep exploratory units as compact executable
   outlines. Add frozen protocols, broad manifests, replication, and hardened
   runtime contracts only after a pilot survives its discriminating control.
9. **Close the routing fan-out once.** After evidence closes, update the owning
   unit/results, its experiment index, and the current decision or compass when
   the route changed. Refresh project memory only when continuation state
   changed. Create a handoff only for an actual session or machine transfer and
   treat it as consumed transport, not a durable current-state owner. Link
   authority surfaces rather than copying the same prose into each one.
10. **Route discoveries.** Promote only stable current behavior into `docs/`; use OpenSpec only for stable compatibility-sensitive contracts.
11. **Verify the boundary.** Check source counts, manifest rows, tracked file set,
   local links, YAML Ain't Markup Language (`YAML`) frontmatter where used, and
   ignored-file behavior before reporting.

## Union Collection Pattern

Use when the user wants all scattered Markdown gathered before cleanup or migration.

- Start from the exact checkout and worktree set.
- Include tracked branch-head Markdown plus dirty or untracked Markdown when the user asks for the union of knowledge.
- Classify at least:
  - `content_present_elsewhere`: identical content already exists at another
    preserved path;
  - `new_content_new_path`: both content and destination path are new;
  - `same_path_divergent_new_content`: the path already exists but the incoming
    content differs and must be preserved separately;
  - `same_path_identical`: both path and content match an existing record.
- Preserve raw snapshots as provenance, including trailing whitespace if it belongs to the original source.
- Write a manifest that records source worktree, branch, path, content hash, classification, and snapshot path.

Do not validate curated style against raw intake snapshots. Validate only that the manifest and snapshot boundary are accurate.

## Open Knowledge Format (`OKF`)-Style Research Migration

Use a repo-native structure rather than depending on app-specific wiki syntax. Durable value comes from typed reading paths, clear authority boundaries, and linked evidence.

Default hierarchy:

```text
research/index.md
research/ideas/index.md
research/investigations/index.md
research/mechanisms/index.md
research/archive/index.md
research/ideas/<slug>/
research/investigations/<slug>/experiments/<unit-id>/unit.md
```

Conventions:

- router `index.md` files may be plain;
- non-router notes may use light frontmatter such as `type: idea`, `type: investigation`, or `type: mechanism`;
- expand and define every abbreviation, shortened arm/hypothesis name, metric
  symbol, dataset/model alias, and coined mechanism name at first use or in a
  terminology registry;
- replace an acronym with a behavior-level plain-language name when expansion
  alone still leaves the mechanism or objective unclear;
- avoid premature `conclusion.md` unless the research direction is actually closed;
- use `outputs/research/<investigation>/<unit-id>/<run-id>/` for executed
  artifacts; record the resolved absolute root and receipts in the unit;
- treat a path as a provenance handle, not proof of execution or metric validity;
- update `docs/AGENT_INDEX.md` and `docs/catalog.yaml` only when `research/` discoverability or authority boundaries change.

If `/research/` is ignored, stage only the intended tracked set with explicit paths or `git add -f` after review. Do not sweep unrelated ignored local research artifacts into validation or commits.

## Supervisor And Independent Advanced-Model Packet Mode

Use when the user needs many heavy diagnostics compressed for a supervisor or
collaborator, or when a conclusion-critical research fork needs judgment from
an independent advanced-model reviewer. Here, an independent advanced-model
reviewer means an advanced model or agent outside the primary implementation
and internal-audit chain. A named model such as Pro or Fable is only one
possible reviewer, never a fixed dependency.

Read [Research Graph and Unit Contract](references/research-graph-contract.md)
and apply its independent advanced-model review gate. Run the cheapest
discriminating observation first; use external review only for a bounded
method, interpretation, promotion, or costly route decision that remains open.

Good default packet:

```text
01_decision_brief.md
02_evidence_atlas.md
03_review_questions.md
```

Keep the packet decision-focused and follow the reference for evidence scope,
blindness, verdicts, and closeout. Reviewer output is advisory and does not
become scientific evidence, originating intent, or project authority.

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

For new research units, also check frontmatter lifecycle fields, outline-or-
protocol/result separation, evidence-tier-appropriate artifact attribution,
and that the bounded verdict does not exceed its evidence scope. Require an
immutable run identifier once execution artifacts need stable comparison; do
not turn a pre-execution exploratory outline into a manifest exercise.

For a closed unit, also check that the experiment index does not retain an old
`planned`, `ready`, or `running` state; that the active decision surface does not
point to a superseded route; and that project memory or a durable handoff does
not contradict the owning result.

## Output Contract

Report:

- source boundary and evidence date window;
- created or updated research/history files, plus any legacy `progress/` sources
  migrated or intentionally left untouched;
- what stayed raw provenance versus synthesized interpretation;
- authority caveats and current-behavior docs/specs touched or intentionally untouched;
- verification commands and scope;
- unresolved evidence gaps and next continuation seeds.
- for executed units, logical/resolved `outputs/research/` roots, receipt state,
  evidence status, architecture-promotion status, and any explicitly
  unauthorized implementation work.
