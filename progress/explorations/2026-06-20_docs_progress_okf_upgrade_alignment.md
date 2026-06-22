---
doc_id: progress.explorations.docs-progress-okf-upgrade-alignment-2026-06-20
layer: progress
doc_type: alignment-decision
status: active-decision
domain: repo-knowledge
summary: Alignment decisions for the docs/progress OKF-style upgrade after collecting Markdown knowledge from linked worktrees, including the approved ideas/investigations/mechanisms convention.
tags: [docs, progress, research, okf, knowledge-graph, worktree-union]
updated: 2026-06-20
---

# Docs/Progress OKF Upgrade Alignment

Date: 2026-06-20

Scope: alignment decision after collecting Markdown knowledge from linked
CoordExp worktrees into
`docs/history/worktree-union/2026-06-20/`.

This note records the direction for the next implementation roadmap. It does
not perform the migration.

## Decision

Rename the current top-level `progress/` knowledge base to `research/` during
the OKF-style migration.

Keep these surfaces separated:

- `docs/`: source of truth for coding, architecture, infrastructure, operator
  workflows, and maintained current behavior documentation.
- `openspec/`: source of truth for stable compatibility-sensitive module and
  service contracts, dependency semantics, schemas, artifacts, metrics, and
  training/evaluation interfaces. Do not use OpenSpec for every ordinary
  dependency edge.
- `research/`: the renamed former `progress/` corpus and the primary research
  knowledge base; tracks new ideas, investigations, mechanisms, hypotheses,
  explorations, experiments, diagnostics, benchmarks, evidence when available,
  negative results, decisions, follow-ups, and interpretation.
- `docs/history/`: non-normative provenance, raw intakes, archived plans, and
  superseded documentation history.

The main refactoring scope is the former `progress/` corpus, not a broad rewrite
of current `docs/` or `openspec/`.

Architecture review files imported from the dirty root are temporary artifacts.
They may be used as input while comparing architecture opinions, but they are
not a durable first-class documentation surface and should be removed or folded
into a compact conclusion after their value is extracted.

Use the following approved `research/` convention:

```text
research/
  index.md
  ideas/
  investigations/
  mechanisms/
  archive/
```

Use the top-level folders by role, not by worktree name:

- `ideas/`: new algorithmic or research directions that may become CoordExp
  capabilities. An idea is allowed to start vague, move through ChatGPT draft
  intake, Codex debate, worktree implementation, training/ablation,
  inference/eval, post-analysis, and then end as promoted, deprecated,
  inconclusive, or absorbed. An idea does not need completed training,
  completed eval, or a finished conclusion to deserve `research/ideas/` if it
  is a valid innovation direction.
- `investigations/`: bounded analysis, ablation, diagnosis, checkpoint-surgery,
  trained-model behavior study, or post-analysis work. Investigations may
  generate important findings without being brand-new innovation directions.
- `mechanisms/`: reusable semantic explanations, failure modes, behavior
  models, and cross-cutting concepts that connect multiple ideas and
  investigations.
- `archive/`: raw imported material, superseded fragments, deprecated plans,
  temporary files, and material that should not remain on the main reading
  path.

Use this default idea layout when an idea needs all lifecycle stages:

```text
research/ideas/<idea-slug>/
  index.md
  overview.md
  draft.md
  discussion.md
  implementation.md
  experiments/
  conclusion.md
  archive/
```

Use this default investigation layout:

```text
research/investigations/<investigation-slug>/
  index.md
  overview.md
  experiments/
  findings.md
  archive/
```

Store reusable mechanism documents directly under `research/mechanisms/` unless
a mechanism needs a larger subfolder:

```text
research/mechanisms/<mechanism-slug>.md
```

Keep the required OKF type values minimal:

```yaml
type: idea | investigation | mechanism
```

Do not create durable types for ordinary note roles such as draft, discussion,
implementation, experiment, finding, repair, launch-health, synthesis, or
conclusion. Use filenames, headings, tags, and Markdown links for those roles.
If state is useful, limit it to optional `state: draft | active | closed` on
`overview.md`; if closed, optionally record `outcome: promoted | deprecated |
inconclusive | absorbed`.

## Lifecycle Routing

Future agents should route the daily research routine as follows:

- ChatGPT Web draft or vague algorithm idea:
  `research/ideas/<idea-slug>/draft.md`.
- Codex attachment, grill-me debate, objections, variants, and approval logic:
  `research/ideas/<idea-slug>/discussion.md`.
- Worktree path, branch, super-power plan, implementation handles, and code
  entrypoints:
  `research/ideas/<idea-slug>/implementation.md`.
- Training runs, ablations, inference/eval, post-analysis, and artifacts that
  test an idea:
  `research/ideas/<idea-slug>/experiments/<date>_<run-or-question>.md`.
- Final verdict for an idea:
  `research/ideas/<idea-slug>/conclusion.md`, with `overview.md` updated.
- Diagnosis-based worktrees, checkpoint surgery, behavior probes, standalone
  ablations, and analysis programs:
  `research/investigations/<investigation-slug>/`.
- Reusable explanation extracted from one or more ideas/investigations:
  `research/mechanisms/<mechanism-slug>.md`.
- Raw, superseded, deprecated, or imported material that should not be the
  first reading path:
  the nearest `archive/`.

A worktree is an execution container, not automatically a research atom. Map it
to an idea only when it represents a new direction that may become a capability;
map it to an investigation when it is analysis, ablation, diagnosis, surgery, or
post-analysis.

Current example routing:

- `prefix-denoising-sft`, `coord-repel-conservative-design`,
  `loss-only-instance-enumeration`, and `row-conditioned-visual-coverage` are
  valid `research/ideas/` entries. Some of them may still be incompletely
  trained, unevaluated, or missing final conclusions; that incompleteness is a
  lifecycle state, not a reason to demote them to investigations or archive.
- `mechanistic-diagnosis-experiments` and
  `autoregressive-binding-template-study` map to `research/investigations/`.
- `fully-compact-2x2-ablation` should attach under a parent idea's
  `experiments/` if it clearly tests that idea; otherwise it can become a small
  investigation.
- `segment-aware-packing-infra` and `codebase-refactoring-program` remain
  `docs/`/`openspec/` first unless they contain research interpretation worth
  preserving.

## Rationale

`progress/` has grown beyond "progress" in the narrow sense. Its durable role is
research memory and semantic knowledge management: ideas, investigations,
mechanisms, hypotheses, explorations, experiments, diagnostics, benchmark
evidence, negative results, design lineage, launch-health notes, and handoff
context. The name `research/` is narrower than `knowledge/`, so it avoids
blurring the boundary with current docs while still describing the corpus
accurately.

The future hierarchy should not over-weight evidence as the primary organizing
axis. Not every valuable research exploration produces solid measured evidence,
and separate worktrees do not all correspond to valid conclusions. Exploratory
work may still preserve high-value ideas, mechanisms, failure narratives,
design alternatives, or future research seeds. The migration should retain that
semantic content without forcing every record through a heavy evidence or status
taxonomy.

The OKF standard remains the structural target: a hierarchical Markdown bundle
with frontmatter-bearing Markdown documents, reserved `index.md` files for
progressive disclosure, optional `log.md` update histories, and standard
Markdown links between documents. OKF's minimal required producer-defined field
is `type`; CoordExp should keep that schema deliberately small. Prefer the
ideas/investigations/mechanisms split, a few durable type values, optional tags,
and standard links over many subclasses that make the knowledge base hard to
maintain or extend.

Keeping `docs/`, `openspec/`, and `research/` separate preserves the existing
CoordExp authority model:

- coding, architecture, infrastructure, and operator truth remains in `docs/`;
- stable module/service contracts and compatibility-sensitive dependency
  semantics remain in `openspec/`;
- research development, semantic mechanisms, empirical evidence, exploration
  lineage, and interpretation move through `research/ideas/`,
  `research/investigations/`, and the reusable `research/mechanisms/` graph;
- raw branch/document intake remains non-normative under `docs/history/`.

The architecture review files are useful as temporary comparison inputs, but
promoting them into a permanent `docs/architecture/reviews/` surface would
create another knowledge lane before the OKF migration has a stable shape.

## Consequence

The implementation roadmap should prioritize:

1. Triage the worktree-union manifest.
2. Reschedule the future `research/` hierarchy around `ideas/`,
   `investigations/`, `mechanisms/`, and `archive/`.
3. Define the OKF-conformant `research/` schema with the minimal required types
   `idea`, `investigation`, and `mechanism`, plus optional tags and links.
4. Pilot migration on a representative, graph-rich `progress/` slice.
5. Rename `progress/` to `research/` and update routing references.
6. Promote or synthesize missing research notes from the raw intake.
7. Keep divergent current-doc snapshots as extraction sources, not overwrite
   inputs.
8. Keep branch `openspec/changes/` snapshots historical unless a contract
   change is explicitly re-approved.
9. Remove or collapse temporary architecture review artifacts after extracting
   durable conclusions.
10. Add validation for OKF conformance, frontmatter, links, catalogs, duplicate
    IDs, the small type vocabulary, and retrieval probes.

Apply these migration guardrails:

- Treat `index.md` files as router/progressive-disclosure files, not semantic
  atoms.
- Treat existing `progress/` folders as source pools, not target folders; do
  not migrate `directions/`, `diagnostics/`, `benchmarks/`, `audits/`, or other
  old production-mode folders one-to-one.
- Create experiment documents for meaningful questions, run families,
  checkpoint comparisons, ablation conclusions, or eval interpretations, not
  for every raw launch.
- Extract `research/mechanisms/` documents only when the mechanism is reusable
  across multiple ideas or investigations, or when it repeatedly serves as
  explanatory glue.
- Do not create a `research/` folder only because a worktree exists. Route
  implementation-only or infra-only worktrees to `docs/`, `openspec/`, or
  archive unless they contain durable research reasoning.
- When an idea closes, update `conclusion.md` and `overview.md`; do not require
  a large final report unless the evidence is complex.
- Promotion into `main` should update `docs/` for current behavior and
  `openspec/` only for stable compatibility-sensitive contracts. Keep
  exploratory history in `research/`.
- Preserve raw worktree intake under `docs/history/worktree-union/` until it is
  intentionally merged into a reading path or archived as provenance.
- Keep validation lightweight: required `type` for non-router research
  documents, allowed values `idea`, `investigation`, and `mechanism`, stale
  folder-reference checks, feasible broken-link checks, and duplicate obvious
  slug checks.

Apply these OKF compatibility guardrails:

- Treat `research/` as an OKF-style knowledge bundle inside the repo: a
  directory hierarchy of Markdown files with YAML frontmatter on non-reserved
  concept documents.
- Keep `index.md` and `log.md` reserved. `index.md` is for progressive
  disclosure and directory listing; `log.md` is optional chronological update
  history. Do not use either as the semantic knowledge atom.
- Every non-reserved `.md` file under `research/` should have parseable YAML
  frontmatter with a non-empty `type`.
- Keep `type` producer-defined and small: `idea`, `investigation`, or
  `mechanism`. Do not introduce a central schema registry or many subclasses.
- Prefer normal Markdown links for graph edges. The surrounding prose explains
  the relationship; do not require typed relation frontmatter in the first
  migration.
- Use structural Markdown headings, lists, tables, and fenced blocks in bodies
  when they improve agent retrieval and human reading.
- Keep citations or provenance handles in the body when a claim depends on
  source docs, artifact roots, configs, commits, metrics, or external material.
- Validators should treat missing optional fields, unknown extension keys, and
  temporarily broken links as soft issues, not fatal bundle failures.

Use this super-power roadmap handoff:

- Write the migration design/roadmap to
  `docs/superpowers/specs/2026-06-20-research-okf-migration-design.md`.
- Write the executable implementation checklist to
  `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md`.
- Let the roadmap cover the full migration architecture, but keep the first
  implementation phase limited to the side-by-side
  `research/ideas/prefix-denoising-sft/` pilot. Do not mass-rename
  `progress/` in the first phase.
- Treat the pilot as accepted when a future agent can open
  `research/ideas/prefix-denoising-sft/overview.md` and reconstruct the
  original idea, debate/approval logic, worktree/branch/code handles,
  implementation state, experiment/artifact handles, incomplete lifecycle
  state, and next action.
- Synthesize from `progress/` and `docs/history/worktree-union/`; link back to
  sources; preserve raw intake; avoid treating file moves as the main migration
  method.
- Defer the lightweight validator unless it is trivial in the pilot. When
  added, it should check only reserved files, required `type`, allowed type
  values, stale folder names, duplicate obvious slugs, and feasible broken
  links.
- Do not create an OpenSpec change unless the migration changes stable config
  schemas, artifact names, metric meanings, training/eval behavior, or operator
  workflows.
- Preserve unrelated dirty work and stage only docs/history/research migration
  files when committing this branch.

Do not create a new OpenSpec change only for the knowledge-base migration unless
the implementation changes normative config schemas, artifact names, metric
semantics, training/eval behavior, or stable operator workflows.

Do not use `research/` to answer current coding, architecture, infrastructure,
or operator behavior when `docs/` or `openspec/` already define it. Do not use
`docs/` as the place for experiment interpretation. Do not use `openspec/` for
ordinary research notes, transient implementation plans, or unstable branch
ideas.

## Evidence

- Scope: `none-yet`
- Raw intake:
  `docs/history/worktree-union/2026-06-20/manifest.tsv`
- Raw intake router:
  `docs/history/worktree-union/2026-06-20/README.md`
- Current docs authority:
  `docs/AGENT_INDEX.md`
- Current machine catalog:
  `docs/catalog.yaml`

## Open Follow-Up

Create the implementation roadmap after this alignment note:

- migration phases;
- row triage policy for the raw intake;
- future `research/ideas/` and `research/investigations/` pilot policy and
  minimum viable semantic grouping policy;
- `research/` frontmatter, relation conventions, minimal type vocabulary, and
  lightweight idea state/outcome fields;
- router/catalog update plan;
- validator and retrieval-probe plan;
- cleanup policy for temporary architecture review artifacts.
