# Research OKF Migration Design

## Objective

Design the staged migration from the current `progress/` research-history
corpus to an OKF-style `research/` knowledge bundle, starting with a
side-by-side pilot for `prefix-denoising-sft`.

The design must preserve raw provenance, keep `docs/` and `openspec/` authority
boundaries intact, and stop at `ready for user approval` before any migration
implementation begins.

## Source Of Truth

- Alignment decision:
  `progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md`
- Raw worktree intake:
  `docs/history/worktree-union/2026-06-20/manifest.tsv`
- Current docs authority:
  `docs/AGENT_INDEX.md`, `docs/catalog.yaml`
- Current progress router:
  `progress/README.md`, `progress/index.yaml`
- OKF upstream references checked on 2026-06-20:
  - `https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf`
  - `https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md`

## Scope

This roadmap covers the whole migration architecture, but the first
implementation phase is intentionally narrow:

```text
research/ideas/prefix-denoising-sft/
```

The first phase creates a side-by-side `research/` pilot. It does not rename or
delete `progress/`, and it does not migrate all diagnostics.

## Non-Goals

- Do not install the OKF `enrichment-agent` package.
- Do not vendor the upstream OKF `SPEC.md` into normative CoordExp docs.
- Do not add `okf_version` frontmatter to `research/index.md` in phase 1; keep
  every `index.md` frontmatter-free for simpler local validation.
- Do not create an OpenSpec change unless stable config schemas, artifact
  names, metric meanings, training/eval behavior, or operator workflows change.
- Do not treat file moves as the migration strategy.
- Do not create empty lifecycle files just to match a template.
- Do not copy artifact trees, logs, metrics JSONL, plots, model outputs, or
  checkpoint files into `research/`.
- Do not begin implementation until the user explicitly approves this roadmap
  and plan.

## Adopted OKF Rules

CoordExp should use OKF as a rules-and-philosophy layer, not as a dependency.
The future `research/` tree is an OKF-style bundle inside the repository.

Adopt these rules:

- `research/` is a directory tree of Markdown files.
- Non-reserved Markdown files have YAML frontmatter.
- Every non-reserved file has a non-empty `type`.
- Reserved `index.md` files are router/progressive-disclosure files and contain
  no frontmatter.
- Reserved `log.md` files are optional chronological update histories.
- Normal Markdown links are the primary graph edge.
- Link semantics live in surrounding prose, not typed relation metadata.
- Unknown extension keys and missing optional fields are allowed.
- Broken links are soft issues during partial migration.

Phase 1 deliberately uses a stricter local convention than upstream OKF by
keeping all `index.md` files frontmatter-free and omitting `okf_version`.

Do not introduce a central schema registry. Use only the local producer-defined
type values:

```yaml
type: idea | investigation | mechanism
```

`type` denotes the top-level semantic bucket. Document role is carried by
filename, headings, tags, and links, not by extra values such as
`type: experiment`, `type: note`, or `type: archive`.

## Target Research Hierarchy

```text
research/
  index.md
  ideas/
  investigations/
  mechanisms/
  archive/
```

### `ideas/`

New algorithmic or research directions that may become CoordExp capabilities.
An idea can start as a ChatGPT Web draft and later move through Codex debate,
worktree implementation, training, ablation, inference/eval, post-analysis, and
promotion or deprecation.

An idea does not need complete training, complete eval, or a final conclusion to
deserve a home under `research/ideas/`.

Default layout when the files are meaningful:

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

Do not create empty files. If there is no conclusion, omit `conclusion.md` and
state the incomplete lifecycle in `overview.md`.

### `investigations/`

Bounded analysis, ablation, diagnosis, checkpoint-surgery, trained-model
behavior study, or post-analysis work.

Default layout:

```text
research/investigations/<investigation-slug>/
  index.md
  overview.md
  experiments/
  findings.md
  archive/
```

### `mechanisms/`

Reusable semantic explanations, failure modes, behavior models, or cross-cutting
concepts that connect multiple ideas and investigations.

Default layout:

```text
research/mechanisms/<mechanism-slug>.md
```

Use a subfolder only when the mechanism becomes too large for one document.

### `archive/`

Raw imported material, superseded fragments, deprecated plans, temporary files,
and material that should not remain on the main reading path.

## First Pilot: `prefix-denoising-sft`

The first pilot proves the `ideas/` lifecycle on a real active research effort.
It should synthesize useful content from current and imported sources into a
small reading path:

```text
research/ideas/prefix-denoising-sft/
  index.md
  overview.md
  draft.md
  discussion.md
  implementation.md
  experiments/
  archive/
```

Do not create `conclusion.md` in the first pilot because the idea is not fully
concluded.

Create `research/ideas/prefix-denoising-sft/experiments/index.md` as the
progressive-disclosure router for experiment notes.

### Required Source Files

- `progress/directions/prefix_denoising_sft_v1.md`
- `progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md`
- `progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md`
- `progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md`
- `docs/history/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md`
- `docs/history/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md`
- `docs/history/worktree-union/2026-06-20/manifest.tsv`
- `docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md`
- `docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md`

Implementation may discover additional sources, but these are the required
minimum.

### Pilot Acceptance

The pilot is successful when a future agent can open:

```text
research/ideas/prefix-denoising-sft/overview.md
```

and reconstruct:

- original idea and motivation;
- debate and approval logic;
- worktree, branch, and code handles;
- implementation state;
- experiment and artifact handles;
- incomplete lifecycle state;
- next action.

## Migration Mechanics

Use synthesis-first migration:

- Read old `progress/` and raw intake material.
- Synthesize the useful content into the new `research/` reading path.
- Link back to sources for provenance.
- Preserve `docs/history/worktree-union/` raw snapshots.
- Avoid one-to-one file moves from old top-level folders.

Existing `progress/` folders are source pools, not target folders. Do not map
`directions/`, `diagnostics/`, `benchmarks/`, `audits/`, or other old
production-mode folders directly into `research/`.

## Routing And Authority

- Current coding, architecture, infrastructure, and operator truth remains in
  `docs/`.
- Stable compatibility-sensitive contracts remain in `openspec/specs/`.
- Research development, source-linked interpretation, negative results,
  historical reasoning, and provenance handles move through `research/`.
- Raw branch and worktree intake remains non-normative under `docs/history/`.

Promotion into `main` should update `docs/` for current behavior and
`openspec/` only for stable contracts. Exploratory history remains in
`research/`.

## Validation Strategy

The first pilot can rely on local one-off checks. A committed validator is a
later phase unless it is trivial.

Lightweight checks should cover:

- `index.md` and `log.md` are reserved and do not carry semantic frontmatter.
- Non-reserved research Markdown files have parseable YAML frontmatter.
- Non-reserved research Markdown files have `type: idea`,
  `type: investigation`, or `type: mechanism`.
- Stale old target names such as `projects/`, `concepts/`, `threads/`,
  `lines/`, `programs/`, and `new_idea` are absent from the generated
  `research/` tree and routing docs, except when roadmap packets mention them
  as rejected alternatives.
- Obvious duplicate slugs are absent.
- Feasible broken-link checks are run. Planned intra-pilot links and required
  source/provenance handles are hard failures; links to not-yet-migrated future
  research content are soft warnings during partial migration.
- No non-router Markdown file under `archive/` should invent `type: archive`.
  Archive is a location/status, not a type. If archived Markdown is kept under
  `research/`, it should retain the nearest semantic type; otherwise raw
  provenance should stay linked from `docs/history/`.

## Review Convergence

Mode: `docs/spec/plan`.

Allowed mutation: documentation/knowledge-artifact-only. Do not create the
first `research/` pilot until the user approves implementation.

Review lanes for this roadmap:

- OKF conformance and naming lane.
- CoordExp docs/governance lane.
- Pilot source coverage and lifecycle lane.
- Implementation-plan executability lane.

Stop state for this design packet: `ready for user approval`.

## Risks And Mitigations

- Risk: the migration becomes a file shuffle.
  Mitigation: pilot output is judged by reading-path quality and source links,
  not by number of files moved.
- Risk: `research/` competes with `docs/`.
  Mitigation: preserve authority boundaries and route current behavior to
  `docs/`.
- Risk: the hierarchy becomes over-classified.
  Mitigation: keep only `ideas/`, `investigations/`, `mechanisms/`, and
  `archive/`, with only three `type` values.
- Risk: incomplete ideas look invalid.
  Mitigation: `overview.md` records incomplete lifecycle state; lack of
  conclusion is valid for active ideas.
- Risk: OKF dependency creep.
  Mitigation: use upstream OKF by reference and local checks only in phase 1.
