# Research OKF Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a side-by-side OKF-style `research/` pilot for `prefix-denoising-sft` while preserving `progress/`, raw worktree intake, and CoordExp docs/OpenSpec authority boundaries.

**Architecture:** Build the new `research/` tree as a synthesized reading path, not a file-move mirror of old `progress/` folders. Keep `index.md` files as routers, put typed content in named Markdown files, and link every synthesized claim back to progress, history, configs, artifacts, commits, or worktree handles.

**Tech Stack:** Markdown, YAML frontmatter, git, `rg`, Python standard library plus installed `yaml` parser, existing CoordExp docs routers, existing docs/history worktree-union intake.

---

## Source Of Truth

- Design spec: `docs/superpowers/specs/2026-06-20-research-okf-migration-design.md`
- Alignment decision: `progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md`
- OKF upstream spec checked 2026-06-20: `https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md`
- Raw intake manifest: `docs/history/worktree-union/2026-06-20/manifest.tsv`

## Mutation And Approval Boundary

This plan is not approved for implementation until the user explicitly says to
start implementation. Before that approval, do not create `research/` migration
files, do not rename `progress/`, and do not update routing to point at
uncreated files.

When implementation is approved, mutate documentation and knowledge artifacts
only. Allowed roots and files are `research/`, `docs/history/worktree-union/`,
`docs/history/README.md`, `docs/AGENT_INDEX.md`, `docs/catalog.yaml`,
`progress/index.yaml`, and the alignment note only if a review finding requires
clarification. Do not edit code, configs, training artifacts, model outputs,
OpenSpec contracts, or external package dependencies.

Repository hygiene note: `.gitignore` intentionally ignores `/research/` as a
hard local-only workspace root. Do not silently weaken that artifact guardrail
inside this pilot. If the pilot is staged or committed, force-add only the
planned Markdown files under `research/` and verify those files are staged; do
not rely on plain `git status --short -- research` before force-add.
The raw worktree-union intake is also part of the migration packet because the
research docs use it as provenance. Do not commit `research/` links that point
to untracked `docs/history/worktree-union/` files.

## Planned Files

Create during implementation:

- `research/index.md`
- `research/ideas/index.md`
- `research/investigations/index.md`
- `research/mechanisms/index.md`
- `research/archive/index.md`
- `research/ideas/prefix-denoising-sft/index.md`
- `research/ideas/prefix-denoising-sft/overview.md`
- `research/ideas/prefix-denoising-sft/draft.md`
- `research/ideas/prefix-denoising-sft/discussion.md`
- `research/ideas/prefix-denoising-sft/implementation.md`
- `research/ideas/prefix-denoising-sft/experiments/index.md`
- `research/ideas/prefix-denoising-sft/experiments/2026-06-14_launch_health_and_branch_isolation.md`
- `research/ideas/prefix-denoising-sft/experiments/2026-06-16_axis_sort_negative_result.md`
- `research/ideas/prefix-denoising-sft/experiments/2026-06-17_inert_objective_root_cause.md`
- `research/ideas/prefix-denoising-sft/archive/index.md`

Modify during implementation:

- `docs/history/README.md`
- `docs/AGENT_INDEX.md`
- `docs/catalog.yaml`
- `progress/index.yaml` only if the alignment-note count check fails
- `progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md` only if review findings require a clarification

Include as raw provenance intake:

- `docs/history/worktree-union/README.md`
- `docs/history/worktree-union/2026-06-20/README.md`
- `docs/history/worktree-union/2026-06-20/manifest.tsv`
- `docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md`
- `docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md`

The full dated intake bundle under `docs/history/worktree-union/2026-06-20/`
may be staged as a single raw provenance unit if the commit policy accepts the
union archive. At minimum, the files above must be tracked before committing
the research docs that cite them.

Do not create `research/ideas/prefix-denoising-sft/conclusion.md` in the pilot.

## Task 1: Preflight Source Inventory

**Files:**
- Read: `progress/directions/prefix_denoising_sft_v1.md`
- Read: `progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md`
- Read: `progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md`
- Read: `progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md`
- Read: `docs/history/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md`
- Read: `docs/history/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md`
- Read: `docs/history/worktree-union/2026-06-20/manifest.tsv`
- Read: `docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md`
- Read: `docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md`
- Read: `progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md`

- [ ] **Step 1: Confirm all required source files exist**

Run:

```bash
python - <<'PY'
from pathlib import Path
paths = [
    "progress/directions/prefix_denoising_sft_v1.md",
    "progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md",
    "progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md",
    "progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md",
    "docs/history/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md",
    "docs/history/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md",
    "docs/history/worktree-union/2026-06-20/manifest.tsv",
    "docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md",
    "docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md",
    "progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md",
]
missing = [p for p in paths if not Path(p).is_file()]
if missing:
    raise SystemExit("missing required source files:\n" + "\n".join(missing))
for p in paths:
    print(f"ok {p}")
PY
```

Expected: every path prints with `ok`.

- [ ] **Step 2: Extract the source map for `overview.md`**

Create a local scratch list in the implementation notes, not a committed file,
with these fields for each source:

```text
source path
role in pilot
claim types to extract
artifact roots or branch handles
superseded-by relationship if any
```

Expected source roles:

```text
progress/directions/prefix_denoising_sft_v1.md
  role: original idea, design rationale, implementation assumptions
progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md
  role: superseded launch smoke and historical caveat
progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md
  role: post-repair launch-health and verification evidence
progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md
  role: review findings and repair motivation
docs/history/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md
  role: historical design spec
docs/history/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md
  role: historical implementation plan
docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md
  role: axis-sort repair negative result
docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md
  role: root-cause analysis and next probes
docs/history/worktree-union/2026-06-20/manifest.tsv
  role: raw worktree-union provenance index; confirms branch, worktree, commit, snapshot hash, same-path status, dirty/untracked status, and snapshot paths for imported prefix-denoising sources
  preserve fields: classification, source_kind, branch, head, worktree, source_path, status, sha256, same_path_in_main, content_paths_in_main, snapshot_path
  required branch filter: branch == codex/prefix-denoising-sft
  required source rows: all rows that are synthesized, cited, or used as implementation context in the pilot
  authority rule: branch OpenSpec, config, and docs snapshot rows are historical branch provenance only; do not present them as current contracts or current behavior
progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md
  role: packet-level migration convention and governance source, not content to synthesize into the prefix-denoising idea
```

## Task 2: Create Root `research/` Routers

**Files:**
- Create: `research/index.md`
- Create: `research/ideas/index.md`
- Create: `research/investigations/index.md`
- Create: `research/mechanisms/index.md`
- Create: `research/archive/index.md`

- [ ] **Step 1: Create `research/index.md`**

Use this content, with no YAML frontmatter:

```markdown
# Research

CoordExp research knowledge is organized as an OKF-style Markdown bundle.

Use this tree for research development, idea lifecycle records, investigations,
mechanism notes, experiment interpretation, negative results, source-linked
interpretation, and provenance handles. Raw worktree intake remains under
`docs/history/worktree-union/` and is linked from `research/`, not copied into
it. Current coding, architecture, infrastructure, and operator behavior remain
in `docs/`. Stable compatibility-sensitive contracts remain in `openspec/`.

## Entry Points

- [Ideas](ideas/) - new research directions that may become CoordExp capabilities
- [Investigations](investigations/) - bounded analysis, ablation, diagnosis, checkpoint surgery, and post-analysis
- [Mechanisms](mechanisms/) - reusable semantic explanations and failure models
- [Archive](archive/) - raw, superseded, deprecated, or non-main-reading-path material

## OKF Conventions

- `index.md` files are routers and contain no frontmatter.
- Non-router Markdown files use YAML frontmatter with `type: idea`, `type: investigation`, or `type: mechanism`.
- `type` denotes the top-level semantic bucket: idea, investigation, or mechanism. Document role is carried by filename, headings, tags, and links.
- Markdown links carry graph edges; surrounding prose explains the relationship.
- Artifact files stay outside `research/`; research docs store handles and interpretation.
```

- [ ] **Step 2: Create subdirectory indexes**

Create `research/ideas/index.md`:

```markdown
# Ideas

New algorithmic or research directions that may become CoordExp capabilities.

An idea can be incomplete. It does not need completed training, completed eval,
or a final conclusion to live here.

## Current Pilots

- [Prefix Denoising SFT](prefix-denoising-sft/)
```

Create `research/investigations/index.md`:

```markdown
# Investigations

Bounded analysis, ablation, diagnosis, checkpoint surgery, trained-model
behavior study, and post-analysis work.

Investigation pilots will be added after the first idea pilot converges.
```

Create `research/mechanisms/index.md`:

```markdown
# Mechanisms

Reusable semantic explanations, failure modes, behavior models, and
cross-cutting concepts that connect multiple ideas and investigations.

Extract mechanism notes only when the explanation is reusable across multiple
research records or repeatedly serves as explanatory glue.
```

Create `research/archive/index.md`:

```markdown
# Archive

Raw imported material, superseded fragments, deprecated plans, temporary files,
and material that should not remain on the main reading path.

Prefer linking to `docs/history/worktree-union/` for raw worktree provenance
instead of copying large source documents here.
```

## Task 3: Create Prefix-Denoising Idea Shell

**Files:**
- Create: `research/ideas/prefix-denoising-sft/index.md`
- Create: `research/ideas/prefix-denoising-sft/archive/index.md`
- Create: `research/ideas/prefix-denoising-sft/experiments/index.md`

- [ ] **Step 1: Create idea router**

Create `research/ideas/prefix-denoising-sft/index.md` with no frontmatter:

```markdown
# Prefix Denoising SFT

## Main Reading Path

- [Overview](overview.md)
- [Draft](draft.md)
- [Discussion](discussion.md)
- [Implementation](implementation.md)
- [Experiments](experiments/)

This idea is active and does not yet have a final `conclusion.md`.
```

- [ ] **Step 2: Create archive router**

Create `research/ideas/prefix-denoising-sft/archive/index.md` with no
frontmatter:

```markdown
# Prefix Denoising SFT Archive

This archive is reserved for material that should not remain on the main
reading path. The first pilot preserves raw worktree intake under
`docs/history/worktree-union/` and links to it rather than copying it here.
```

- [ ] **Step 3: Create experiments router**

Create `research/ideas/prefix-denoising-sft/experiments/index.md` with no
frontmatter:

```markdown
# Prefix Denoising SFT Experiments

## Experiment Notes

- [Launch Health And Branch Isolation](2026-06-14_launch_health_and_branch_isolation.md)
- [Axis-Sort Repair Negative Result](2026-06-16_axis_sort_negative_result.md)
- [Inert Objective Root-Cause Analysis](2026-06-17_inert_objective_root_cause.md)
```

## Task 4: Write `overview.md`

**Files:**
- Create: `research/ideas/prefix-denoising-sft/overview.md`

- [ ] **Step 1: Write OKF frontmatter**

Use this frontmatter:

```yaml
---
type: idea
title: Prefix Denoising SFT
description: Explores whether Stage-1 compact detection teacher forcing can improve coordinate robustness by training on clean and coordinate-noised prefix views.
tags: [stage1, compact-detection, prefix-denoising, teacher-forcing, coordinate-robustness]
state: active
updated: 2026-06-20
---
```

- [ ] **Step 2: Write control-panel sections**

Include these sections in order:

```markdown
# Prefix Denoising SFT

## Current State

## Central Question

## Current Interpretation

## Worktree And Branch Handles

## Main Reading Path

## Key Evidence

## Artifact Handles

## Manifest Provenance

## Source Map

## Next Action

## Sources
```

Required content:

- State that the idea is valid and active, but incompletely trained/evaluated
  and not concluded.
- State that initial V1 launch wiring became healthy after branch-isolation
  repair.
- State that later analysis found the trained objective likely inert for the
  tested compact-coordinate model and that a matched control is needed before a
  final verdict.
- Link to `draft.md`, `discussion.md`, `implementation.md`, and the three
  experiment notes created in later tasks.
- Link back to every required source listed in Task 1.
- Include the full relative snapshot paths, not only hash substrings:
  `docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md`
  and
  `docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md`.
- Include a compact manifest provenance table populated from
  `docs/history/worktree-union/2026-06-20/manifest.tsv` for the
  prefix-denoising rows synthesized, cited, or used as implementation context.
  The table must include these columns:
  `classification`, `source_kind`, `branch`, `head`, `worktree`,
  `source_path`, `status`, `sha256`, `same_path_in_main`,
  `content_paths_in_main`, and `snapshot_path`.
- Label any branch OpenSpec, config, or docs snapshot row as historical branch
  provenance, not a current contract or current behavior source.
- Preserve artifact roots as paths, not copied files.
- End the file with `## Sources`.

## Task 5: Write `draft.md`, `discussion.md`, And `implementation.md`

**Files:**
- Create: `research/ideas/prefix-denoising-sft/draft.md`
- Create: `research/ideas/prefix-denoising-sft/discussion.md`
- Create: `research/ideas/prefix-denoising-sft/implementation.md`

- [ ] **Step 1: Write `draft.md`**

Use frontmatter:

```yaml
---
type: idea
title: Prefix Denoising SFT Draft
description: Original V1 idea and rationale for clean/noisy prefix-denoising Stage-1 compact detection SFT.
tags: [stage1, prefix-denoising, draft]
updated: 2026-06-20
---
```

Required sections:

```markdown
# Prefix Denoising SFT Draft

## Motivation

## V1 Shape

## Accepted Constraints

## Non-Goals

## Sources
```

Synthesize from `progress/directions/prefix_denoising_sft_v1.md` and
`docs/history/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md`.
End the file with `## Sources`, citing both exact source paths.

- [ ] **Step 2: Write `discussion.md`**

Use frontmatter:

```yaml
---
type: idea
title: Prefix Denoising SFT Discussion
description: Debate, audit findings, repair decisions, and unresolved interpretation boundaries for prefix-denoising SFT V1.
tags: [stage1, prefix-denoising, discussion, audit]
updated: 2026-06-20
---
```

Required sections:

```markdown
# Prefix Denoising SFT Discussion

## Approval Logic

## Audit Findings

## Repair Decisions

## Interpretation Boundaries

## Sources
```

Synthesize from `progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md`,
`progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md`,
and `docs/history/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md`.
End the file with `## Sources`, citing all exact source paths used.

- [ ] **Step 3: Write `implementation.md`**

Use frontmatter:

```yaml
---
type: idea
title: Prefix Denoising SFT Implementation
description: Worktree, branch, code-surface, verification, and artifact handles for the prefix-denoising SFT V1 implementation.
tags: [stage1, prefix-denoising, implementation, worktree]
updated: 2026-06-20
---
```

Required sections:

```markdown
# Prefix Denoising SFT Implementation

## Worktree

## Branch

## Code Surfaces

## Config Surfaces

## Verification Handles

## Artifact Handles

## Current Implementation Read

## Sources
```

Record:

```text
worktree: /data/CoordExp/.worktrees/geometry-aware-denoising-sft
branch: codex/prefix-denoising-sft
```

Do not claim the implementation is merged into main unless verified at
implementation time.
End the file with `## Sources`, citing the plan/spec/history sources and any
manifest rows used for implementation provenance.

## Task 6: Write Experiment Notes

**Files:**
- Create: `research/ideas/prefix-denoising-sft/experiments/2026-06-14_launch_health_and_branch_isolation.md`
- Create: `research/ideas/prefix-denoising-sft/experiments/2026-06-16_axis_sort_negative_result.md`
- Create: `research/ideas/prefix-denoising-sft/experiments/2026-06-17_inert_objective_root_cause.md`

- [ ] **Step 1: Write launch-health experiment note**

Use frontmatter:

```yaml
---
type: idea
title: Prefix Denoising Launch Health And Branch Isolation
description: Tiny launch-health and branch-isolation repair evidence for prefix-denoising SFT V1.
tags: [stage1, prefix-denoising, launch-health, tiny]
updated: 2026-06-20
---
```

Required sections:

```markdown
# Prefix Denoising Launch Health And Branch Isolation

## Scope

## Historical Smoke

## Supersession

## Repair Evidence

## Artifact Handles

## Interpretation Limit

## Sources
```

Synthesize from the 2026-06-14 launch-health note and 2026-06-15 repair note.
End the file with `## Sources`, citing both exact source paths and the
manifest handles for the same-path-identical branch-head source rows.

- [ ] **Step 2: Write axis-sort negative-result note**

Use frontmatter:

```yaml
---
type: idea
title: Prefix Denoising Axis-Sort Repair Negative Result
description: Diagnostic negative result showing bbox endpoint sorting improves materialization but does not recover localization quality.
tags: [stage1, prefix-denoising, negative-result, eval]
updated: 2026-06-20
---
```

Required sections:

```markdown
# Prefix Denoising Axis-Sort Repair Negative Result

## Question

## Method

## Result

## Interpretation

## Artifact Handles

## Sources
```

Synthesize from the snapshot path ending in
`2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md`.
End the file with `## Sources`, citing the full snapshot path and manifest
handle for the `688fa6f04eb5` source row.

- [ ] **Step 3: Write inert-objective root-cause note**

Use frontmatter:

```yaml
---
type: idea
title: Prefix Denoising Inert Objective Root-Cause Analysis
description: Root-cause analysis arguing that the tested prefix-denoising objective was inert and baseline comparison was confounded.
tags: [stage1, prefix-denoising, root-cause, eval-validity]
updated: 2026-06-20
---
```

Required sections:

```markdown
# Prefix Denoising Inert Objective Root-Cause Analysis

## Headline

## Evidence

## Interpretation

## Confounds

## Recommended Next Probes

## Sources
```

Synthesize from the snapshot path ending in
`2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md`.
End the file with `## Sources`, citing the full snapshot path and manifest
handle for the `ed610b284219` dirty/untracked source row.

## Task 7: Update Agent Routing

**Files:**
- Modify: `docs/AGENT_INDEX.md`
- Modify: `docs/catalog.yaml`
- Verify: `progress/index.yaml`

- [ ] **Step 1: Update `docs/AGENT_INDEX.md`**

Add `research/` as an experimental side-by-side research knowledge entrypoint.
Preserve the current `progress/` usage rule until the full migration lands.
Update the file frontmatter `updated` date to the implementation date.
Insert a new `## Research Knowledge Pilot` subsection after the existing
`## Historical Docs Usage Rule` section and before `## Suggested Search Seeds`.
Preserve the existing `## Progress Usage Rule` and `## Historical Docs Usage
Rule` text.

Required routing text:

```markdown
## Research Knowledge Pilot

Use [research/](../research/) for the OKF-style idea, investigation, and
mechanism pilot.

During the pilot, `progress/` remains the historical/evidence source of truth
and `research/` is the synthesized reading path.

Do not answer current coding, architecture, infrastructure, operator, schema,
artifact, metric, or training/eval behavior from `research/` when `docs/` or
`openspec/specs/` cover it.
```

- [ ] **Step 2: Update `docs/catalog.yaml`**

Add a machine-readable entrypoint without changing current authority for
`docs/`, `openspec/`, or `progress/`.

Update the top-level `updated` date to the implementation date and add these
exact nested values while preserving existing keys:

```yaml
entrypoints:
  human:
    research: research/index.md
  agent:
    research: research/index.md

authority:
  research_knowledge_pilot: research/
  research_knowledge_pilot_status: pilot_synthesized_reading_path_not_current_behavior_authority
```

If the exact nesting differs in the current catalog, preserve existing keys and
add only the new `research` entries. Do not duplicate top-level `entrypoints`
or `authority` keys.

- [ ] **Step 3: Verify the alignment note remains machine-discoverable**

The alignment note is already registered in `progress/index.yaml`. Verify
exactly one entry exists in `groups.explorations` and exactly one matching note
exists in `notes`; do not add a duplicate. If either count is `0`, add the
missing entry once. If either count is greater than `1`, remove only the
duplicate alignment-note entry while preserving unrelated entries.

Target path:

```text
progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md
```

Required note shape:

```yaml
- path: progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md
  title: Docs/Progress OKF Upgrade Alignment
  kind: alignment-decision
  status: active-decision
  summary: Alignment decision for renaming progress to research, preserving docs and openspec authority boundaries, and adopting the ideas/investigations/mechanisms OKF-style hierarchy.
```

## Task 8: Validate OKF And Docs Hygiene

Phase 1 deliberately uses a stricter local convention than upstream OKF by
keeping all `index.md` files frontmatter-free and omitting `okf_version`.

**Files:**
- Validate: `research/`
- Validate: `docs/AGENT_INDEX.md`
- Validate: `docs/catalog.yaml`
- Validate: `progress/index.yaml`
- Validate: `progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md`

- [ ] **Step 1: Parse YAML frontmatter and reserved indexes**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re
import yaml

allowed = {"idea", "investigation", "mechanism"}
for path in sorted(Path("research").rglob("*.md")):
    text = path.read_text(encoding="utf-8")
    if path.name == "index.md":
        if text.startswith("---\n"):
            raise SystemExit(f"reserved file has frontmatter: {path}")
        continue
    if path.name == "log.md":
        if text.startswith("---\n"):
            raise SystemExit(f"reserved file has frontmatter: {path}")
        bad = [
            line for line in text.splitlines()
            if line.startswith("## ") and not re.match(r"^## \d{4}-\d{2}-\d{2}(?:\b|$)", line)
        ]
        if bad:
            raise SystemExit(f"bad log date heading in {path}: {bad[0]}")
        continue
    if not text.startswith("---\n"):
        raise SystemExit(f"missing frontmatter: {path}")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise SystemExit(f"unterminated frontmatter: {path}")
    data = yaml.safe_load(text[4:end]) or {}
    value = data.get("type")
    if value not in allowed:
        raise SystemExit(f"bad type {value!r}: {path}")
    print(f"ok {path}: type={value}")
PY
```

Expected: every non-router file prints `type=idea`; every `index.md` is
accepted as reserved.

- [ ] **Step 2: Parse router YAML files**

Run:

```bash
python - <<'PY'
import yaml
from pathlib import Path
for path in ["docs/catalog.yaml", "progress/index.yaml"]:
    yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    print(f"{path}: ok")
PY
```

Expected:

```text
docs/catalog.yaml: ok
progress/index.yaml: ok
```

- [ ] **Step 3: Assert router keys are preserved**

Run:

```bash
python - <<'PY'
from pathlib import Path
import yaml

catalog = yaml.safe_load(Path("docs/catalog.yaml").read_text(encoding="utf-8"))
progress = yaml.safe_load(Path("progress/index.yaml").read_text(encoding="utf-8"))

assert catalog["entrypoints"]["human"]["docs"] == "docs/README.md"
assert catalog["entrypoints"]["human"]["progress"] == "progress/README.md"
assert catalog["entrypoints"]["human"]["research"] == "research/index.md"
assert catalog["entrypoints"]["agent"]["router"] == "docs/AGENT_INDEX.md"
assert catalog["entrypoints"]["agent"]["catalog"] == "docs/catalog.yaml"
assert catalog["entrypoints"]["agent"]["progress"] == "progress/index.yaml"
assert catalog["entrypoints"]["agent"]["research"] == "research/index.md"
assert catalog["authority"]["current_truth"] == "docs/"
assert catalog["authority"]["stable_contracts"] == "openspec/specs/"
assert catalog["authority"]["historical_evidence"] == "progress/"
assert catalog["authority"]["research_knowledge_pilot"] == "research/"
assert "not_current_behavior_authority" in catalog["authority"]["research_knowledge_pilot_status"]

target = "progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md"
group_count = progress["groups"]["explorations"].count(target)
note_count = sum(1 for note in progress["notes"] if note.get("path") == target)
assert group_count == 1, f"expected one groups.explorations entry, got {group_count}"
assert note_count == 1, f"expected one notes entry, got {note_count}"

print("router keys: ok")
PY
```

Expected:

```text
router keys: ok
```

- [ ] **Step 4: Assert planned file set, provenance handles, links, and slugs**

Run:

```bash
python - <<'PY'
import csv
from pathlib import Path
import re

pilot_root = Path("research/ideas/prefix-denoising-sft")
manifest_path = Path("docs/history/worktree-union/2026-06-20/manifest.tsv")

planned = {
    "research/index.md",
    "research/ideas/index.md",
    "research/investigations/index.md",
    "research/mechanisms/index.md",
    "research/archive/index.md",
    "research/ideas/prefix-denoising-sft/index.md",
    "research/ideas/prefix-denoising-sft/overview.md",
    "research/ideas/prefix-denoising-sft/draft.md",
    "research/ideas/prefix-denoising-sft/discussion.md",
    "research/ideas/prefix-denoising-sft/implementation.md",
    "research/ideas/prefix-denoising-sft/experiments/index.md",
    "research/ideas/prefix-denoising-sft/experiments/2026-06-14_launch_health_and_branch_isolation.md",
    "research/ideas/prefix-denoising-sft/experiments/2026-06-16_axis_sort_negative_result.md",
    "research/ideas/prefix-denoising-sft/experiments/2026-06-17_inert_objective_root_cause.md",
    "research/ideas/prefix-denoising-sft/archive/index.md",
}
missing = [p for p in sorted(planned) if not Path(p).is_file()]
if missing:
    raise SystemExit("missing planned files:\n" + "\n".join(missing))

source_coverage = {
    "research/ideas/prefix-denoising-sft/overview.md": [
        "progress/directions/prefix_denoising_sft_v1.md",
        "progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md",
        "progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md",
        "progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md",
        "docs/history/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md",
        "docs/history/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md",
        "docs/history/worktree-union/2026-06-20/manifest.tsv",
        "docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md",
        "docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md",
    ],
    "research/ideas/prefix-denoising-sft/draft.md": [
        "progress/directions/prefix_denoising_sft_v1.md",
        "docs/history/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md",
    ],
    "research/ideas/prefix-denoising-sft/discussion.md": [
        "progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md",
        "progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md",
        "docs/history/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md",
    ],
    "research/ideas/prefix-denoising-sft/implementation.md": [
        "docs/history/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md",
        "docs/history/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md",
        "docs/history/worktree-union/2026-06-20/manifest.tsv",
        "/data/CoordExp/.worktrees/geometry-aware-denoising-sft",
        "codex/prefix-denoising-sft",
    ],
    "research/ideas/prefix-denoising-sft/experiments/2026-06-14_launch_health_and_branch_isolation.md": [
        "progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md",
        "progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md",
        "docs/history/worktree-union/2026-06-20/manifest.tsv",
    ],
    "research/ideas/prefix-denoising-sft/experiments/2026-06-16_axis_sort_negative_result.md": [
        "docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md",
        "docs/history/worktree-union/2026-06-20/manifest.tsv",
        "688fa6f04eb57ce5c9d48e0178551e7188281df6150d786818afe4911812b915",
    ],
    "research/ideas/prefix-denoising-sft/experiments/2026-06-17_inert_objective_root_cause.md": [
        "docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md",
        "docs/history/worktree-union/2026-06-20/manifest.tsv",
        "ed610b284219c2e96fe90d2a4c446978b96ededc767c0ce3118417f847846871",
        "worktree-dirty",
        "??",
    ],
}

for file_name, handles in source_coverage.items():
    text = Path(file_name).read_text(encoding="utf-8")
    headings = re.findall(r"^## .+$", text, flags=re.MULTILINE)
    if not headings or headings[-1] != "## Sources":
        raise SystemExit(f"{file_name} must end with a ## Sources section")
    missing_handles = [h for h in handles if h not in text]
    if missing_handles:
        raise SystemExit(f"{file_name} missing source handles:\n" + "\n".join(missing_handles))

overview = Path("research/ideas/prefix-denoising-sft/overview.md").read_text(encoding="utf-8")
implementation = Path("research/ideas/prefix-denoising-sft/implementation.md").read_text(encoding="utf-8")
manifest_surface = overview + "\n" + implementation

required_manifest_columns = [
    "classification",
    "source_kind",
    "branch",
    "head",
    "worktree",
    "source_path",
    "status",
    "sha256",
    "same_path_in_main",
    "content_paths_in_main",
    "snapshot_path",
]
missing_columns = [name for name in required_manifest_columns if name not in manifest_surface]
if missing_columns:
    raise SystemExit("manifest provenance table missing columns:\n" + "\n".join(missing_columns))

with manifest_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
rows_by_source = {row["source_path"]: row for row in rows}
required_manifest_sources = [
    "docs/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md",
    "docs/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md",
    "progress/directions/prefix_denoising_sft_v1.md",
    "progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md",
    "progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md",
    "progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md",
    "progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md",
    "progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md",
]
for source in required_manifest_sources:
    row = rows_by_source.get(source)
    if not row:
        raise SystemExit(f"manifest missing required source row: {source}")
    for key in required_manifest_columns:
        value = row[key]
        if not value or value == "-":
            continue
        if key == "snapshot_path":
            value = "docs/history/worktree-union/2026-06-20/" + value
        if value not in manifest_surface:
            raise SystemExit(f"manifest value missing from overview/implementation: {source} {key}={value}")

if "historical branch provenance" not in manifest_surface:
    raise SystemExit("manifest provenance must label branch OpenSpec/config/docs snapshots as historical branch provenance")

link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
for path in sorted(Path("research").rglob("*.md")):
    text = path.read_text(encoding="utf-8")
    for raw in link_re.findall(text):
        target = raw.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        resolved = (path.parent / target).resolve()
        if resolved.exists():
            continue
        path_resolved = path.resolve()
        try:
            path_resolved.relative_to(pilot_root.resolve())
            in_pilot = True
        except ValueError:
            in_pilot = False
        try:
            resolved.relative_to(pilot_root.resolve())
            target_in_pilot = True
        except ValueError:
            target_in_pilot = False
        if in_pilot and target_in_pilot:
            raise SystemExit(f"broken planned intra-pilot link in {path}: {raw}")
        print(f"warning: soft broken cross-tree or future link in {path}: {raw}")

for parent in [Path("research/ideas"), Path("research/investigations"), Path("research/mechanisms")]:
    if not parent.exists():
        continue
    slugs = [p.stem if p.is_file() else p.name for p in parent.iterdir() if not p.name.startswith(".")]
    dupes = sorted({s for s in slugs if slugs.count(s) > 1})
    if dupes:
        raise SystemExit(f"duplicate slugs under {parent}: {dupes}")

print("research file set, source coverage, manifest provenance, links, slugs: ok")
PY
```

Expected:

```text
research file set, source coverage, manifest provenance, links, slugs: ok
```

- [ ] **Step 5: Check naming residue, whitespace, and scoped diff hygiene**

Run:

```bash
if rg -n "(^|/)(projects|concepts|programs|threads|lines)/|type: (project|note|experiment|concept)\b|\bnew_idea\b" research docs/AGENT_INDEX.md docs/catalog.yaml progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md; then
  echo "unexpected rejected naming residue" >&2
  exit 1
fi
if rg -n "[ \t]$" research docs/AGENT_INDEX.md docs/catalog.yaml progress/index.yaml progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md; then
  echo "unexpected trailing whitespace" >&2
  exit 1
fi
git diff --check -- research docs/AGENT_INDEX.md docs/catalog.yaml progress/index.yaml progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md
git check-ignore -v research/index.md research/ideas/prefix-denoising-sft/index.md || true
```

Expected:

```text
rejected naming residue check exits 0 with no matches
trailing whitespace check exits 0 with no matches
scoped git diff --check exits 0
git check-ignore reports the existing /research/ ignore rule; staging this pilot later requires git add -f for the planned Markdown files
```

## Task 9: Review And Stop For User Approval

**Files:**
- Review: every file changed in this plan

- [ ] **Step 1: Summarize changed files**

Run:

```bash
git status --short -- research docs/history/worktree-union docs/history/README.md docs/AGENT_INDEX.md docs/catalog.yaml progress/index.yaml progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md
git status --ignored --short --untracked-files=all -- research
git status --short
```

Expected: before staging, the ignored scoped status may list the planned
`research/` Markdown files as ignored because `/research/` is intentionally in
`.gitignore`; after staging, use `git add -f` for only the planned Markdown
files under `research/` if a commit is requested. The scoped status should list
only approved research migration files plus approved router files. The global
status may show unrelated dirty work; report it separately as out of scope and
do not clean or revert it. No source code, config, artifact, or OpenSpec
contract files should appear from this implementation.

If the user asks to stage or commit this pilot, stage the research files with an
explicit force-add list and stage the raw worktree-union provenance files that
the research docs cite. Then verify both the synthesized reading path and raw
provenance handles are tracked in the index:

```bash
git add -f -- \
  research/index.md \
  research/ideas/index.md \
  research/investigations/index.md \
  research/mechanisms/index.md \
  research/archive/index.md \
  research/ideas/prefix-denoising-sft/index.md \
  research/ideas/prefix-denoising-sft/overview.md \
  research/ideas/prefix-denoising-sft/draft.md \
  research/ideas/prefix-denoising-sft/discussion.md \
  research/ideas/prefix-denoising-sft/implementation.md \
  research/ideas/prefix-denoising-sft/experiments/index.md \
  research/ideas/prefix-denoising-sft/experiments/2026-06-14_launch_health_and_branch_isolation.md \
  research/ideas/prefix-denoising-sft/experiments/2026-06-16_axis_sort_negative_result.md \
  research/ideas/prefix-denoising-sft/experiments/2026-06-17_inert_objective_root_cause.md \
  research/ideas/prefix-denoising-sft/archive/index.md

git add -- \
  docs/history/README.md \
  docs/history/worktree-union/README.md \
  docs/history/worktree-union/2026-06-20/README.md \
  docs/history/worktree-union/2026-06-20/manifest.tsv \
  docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md \
  docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md

git ls-files --error-unmatch \
  research/index.md \
  research/ideas/prefix-denoising-sft/overview.md \
  docs/history/worktree-union/README.md \
  docs/history/worktree-union/2026-06-20/README.md \
  docs/history/worktree-union/2026-06-20/manifest.tsv \
  docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md \
  docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md
```

The alignment-note registrations in `docs/catalog.yaml` and `progress/index.yaml`
come from the earlier collection/alignment phase. Task 7 verifies their exact
count and only adds the new `research/` routing entries unless that count check
fails.

- [ ] **Step 2: Prepare review packet**

In the final implementation report, include:

```text
mode: docs/spec/plan implementation
mutation scope: documentation and knowledge artifacts only
research pilot path: research/ideas/prefix-denoising-sft/
source files synthesized
verification commands and results
files changed
remaining gate: user review before mass migration or progress rename
```

- [ ] **Step 3: Do not continue into broader migration**

Stop after the pilot. Do not migrate `autoregressive-binding-template-study`,
do not create `research/investigations/` content beyond the index, and do not
rename `progress/` in the same implementation pass.
