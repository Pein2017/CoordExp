---
doc_id: progress.audits.repository-stewardship-2026-05-20
layer: progress
doc_type: audit
status: active-audit
domain: repo
summary: Repository stewardship diagnosis, cleanup candidates, and long-term governance design for CoordExp.
updated: 2026-05-20
---

# Repository Stewardship Audit (2026-05-20)

Scope: whole-repo structure, source/config/script/docs/artifact boundaries, and
future governance. `temp/` was intentionally ignored. `output_remote/` is marked
no-touch for this pass because it is temporary and expected to be deleted.

Evidence commands:

- `rtk git status --short --branch`: clean worktree, `main` ahead of
  `origin/main` by 2 commits before this audit.
- `du -h --max-depth=1 . | sort -h`: largest local roots were
  `public_data/` 136G, `temp/` 23G, `outputs/` 21G, `output_remote/` 12G,
  `external/` 9.9G, model/checkpoint local roots around 8G each, and
  `.codex/` 2.8G.
- `git ls-files | awk -F/ '{print $1}' | sort | uniq -c`: tracked surface is
  mostly `openspec/` 548 files, `configs/` 301, `src/` 285, `tests/` 241,
  `progress/` 197, `scripts/` 81, `docs/` 80, `public_data/` 78.
- `git ls-files -o --exclude-standard`: no unignored files; the allowlist is
  effective.
- `git ls-files -i -o --exclude-standard | awk ...`: ignored local state is
  dominated by `public_data/`, `outputs/`, `output_remote/`, `research/`, and
  agent/runtime caches.

## Executive Diagnosis

The repo already has a strong governance spine:

- `docs/PROJECT_CONTEXT.md` defines authority and read order.
- `docs/catalog.yaml` and `docs/AGENT_INDEX.md` give routing.
- `docs/standards/REPO_HYGIENE.md` defines tracked vs workspace assets.
- `manifests/public_data_provenance/` records reproducible processed data.

The current problem is not a missing structure. The problem is drift around the
structure:

- the root README lagged behind current config paths;
- local runtime state is visible at repo root even when ignored;
- `progress/diagnostics/artifacts/` has begun absorbing generated images,
  per-image tables, and launch logs that should mostly live in `outputs/`;
- some historical planning docs and draft docs are discoverable under `docs/`
  even though they are not current operator truth;
- Codex memory auto-commit scripts conflicted with the current local-only memory
  policy and were broken under the current `.gitignore`.

## Directory Classification

Core tracked assets:

- `src/`: importable runtime library. Current seams are reasonable:
  `config`, `datasets`, `detection`, `training_runtime`, `trainers`, `infer`,
  `eval`, `bootstrap`, `metrics`, `common`, and `vis`.
- `configs/`: YAML-first behavior. Current lifecycle folders are partly clear
  (`prod`, `smoke`, `ablation`, `negative`, `profiles`, `analysis`, `bench`)
  but should be treated as a hard rule going forward.
- `scripts/`: stable entrypoints plus wrappers. The directory is well routed by
  `scripts/README.md`; analysis runners are thin wrappers over `src/analysis`.
- `tests/`: broad contract surface. Large tests exist, especially Stage-2 and
  infer/eval, but the directory is functionally valuable rather than clutter.
- `docs/`: current operator truth and standards. Some historical and active
  implementation drafts need clearer promotion/retirement rules.
- `openspec/`: contract governance. Keep for stable compatibility-sensitive
  contracts only.
- `progress/`: historical evidence. Good category routing exists, but artifact
  retention needs tighter limits.
- `public_data/` tracked code only: converters, pipeline, scripts, and tests.
- `manifests/public_data_provenance/`: small tracked reproducibility records.
- `.codex/skills/`: tracked repo-local agent capability surface.
- `ops/`: workstation and agent policy helpers, not CoordExp pipeline code.

Workspace assets, not tracked:

- `outputs/`: canonical experiment artifact and sync surface.
- `output_remote/`: temporary transitional root, no-touch here, planned deletion.
- `public_data/coco`, `public_data/lvis`, `public_data/vg` raw/processed data.
- `model_cache/`, `model_cache_remote/`, merged model directories.
- `external/`, `.worktrees/`: local checkouts/worktrees.
- `.codex/` runtime state outside skills, `.claude/`, `.gemini/`, `.history/`,
  `.gitnexus/`, `.vendor/`, caches, and credentials.

Scratch or removable local noise:

- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `tb/`, `result/`,
  `monitor_dumps/`, local root token/cookie files, and the stray root file
  `(deleted)`.

## Findings

### P1: Root README had stale training paths

Evidence: `README.md` pointed to `configs/dlora/sft_base.yaml`, but the current
tracked Stage-1 entrypoint is `configs/stage1/sft_base.yaml`, and current
compact detection routes live under
`configs/stage1/recursive_detection_ce_latest/`.

Impact: the first human-facing entrypoint could route a new run into a missing
config family.

Action taken: updated root README layout and training examples to current
Stage-1 and latest compact detection config paths. Also updated
`scripts/analysis/analyze_token_lengths.py` defaults away from the removed
`configs/dlora/` family and away from writing analysis output under `docs/`.

Verification: `rg -n "configs/dlora" README.md docs scripts configs` should
return only historical audit references, if any.

### P1: Codex memory auto-commit ops were stale and broken

Evidence: `ops/codex/commit_codex_memories.sh --dry-run` failed because
`.codex/memories` is ignored. This also contradicted the current local-only
memory policy.

Impact: future agents could accidentally revive noisy source-control coupling
between private runtime state and repository commits.

Action taken: removed the auto-commit/watch/install scripts and replaced
`ops/codex/README.md` with a short policy note.

Verification: `rg -n "commit_codex_memories|watch_codex_memories|memory auto-commit" ops docs .gitignore`.

### P1: Tracked `progress/diagnostics/artifacts/` is becoming an artifact sink

Evidence: tracked files include PNG galleries around 1.2-2.0 MiB each,
per-image JSON tables, duplicate-guard reports, and launch logs under
`progress/diagnostics/artifacts/`. `progress/diagnostics` totals about 41M.

Impact: this turns Git history into a run-artifact store and makes future
clones heavier. The historical evidence layer should carry summaries and
small curated evidence, not full report payloads.

Recommended action: keep `README.md`, summary JSON/MD, and a tiny curated
sample set only when necessary. Move or stop tracking large PNGs, per-image
tables, and raw logs; point to `outputs/...` artifact roots instead.

Verification: after cleanup, `git ls-files progress/diagnostics/artifacts |
xargs du -ch` should stay small, and each artifact cluster should have a
router README with exact output pointers.

### P1: Local asset sprawl is under control for Git, but not for humans

Evidence: ignored local roots include `public_data/` 136G, `outputs/` 21G,
`external/` 9.9G, a merged checkpoint root around 8G, and `.codex/` 2.8G.

Impact: even when Git is clean, repo root is noisy. New contributors and
agents can mistake local runtime state for project structure.

Action taken: hardened `.gitignore` with explicit local-only roots and
credential/runtime patterns, so future allowlist edits do not accidentally
surface these paths.

Recommended action: physically move large optional local roots outside repo
when convenient, or use a predictable local-only prefix such as `.local/` for
future checkouts/caches. Do not move `public_data/` or `outputs/` without a
separate migration plan.

### P2: `docs/` mixes current truth with historical/draft material

Evidence: files under `docs/training/` include canonical pages, historical
notes, and an active implementation draft. `docs/superpowers/` plans/specs are
tracked but not part of `docs/catalog.yaml`.

Impact: `docs/` is supposed to be current operator truth. Historical plans are
useful, but they should not compete with canonical runbooks.

Recommended action: keep stable docs in routed domain pages; move stale design
or implementation plans to `progress/explorations/` or `progress/audits/` after
they stop being active. For active drafts, keep status explicit and add them to
the catalog only when they are expected retrieval targets.

### P2: Config lifecycle is visible but not enforced enough

Evidence: config families already use useful names like `prod`, `smoke`,
`ablation`, `negative`, `analysis`, and `bench`. Some dense clusters, especially
latest compact detection, contain many close variants.

Impact: without a lifecycle rule, every new ablation creates permanent config
surface area.

Action taken: added explicit config lifecycle rules to
`docs/standards/REPO_HYGIENE.md`.

Recommended action: after each experiment cluster concludes, keep the winning
or comparator `prod` config, keep cheap `smoke` configs that still validate an
active surface, keep `negative` only for tests, and move conclusions into
`progress/`.

## Recommended Target Structure

The target structure should remain close to the current repo rather than invent
a new taxonomy:

```text
src/                         importable runtime library
configs/
  _shared/                   snippets and reusable config fragments
  stage1/                    current Stage-1 training configs
  stage2_two_channel/        current Stage-2 operator configs
  infer/ eval/ postop/       YAML-first runtime configs
  bench/ analysis/           report and historical analysis configs
scripts/
  README.md                  entrypoint router
  run_infer.py, ...          stable YAML-first commands
  analysis/                  maintained report/probe wrappers
  tools/                     small deterministic utilities
public_data/                 data tooling plus local ignored data roots
manifests/public_data_provenance/
docs/                        current operator truth and standards
openspec/                    stable compatibility contracts
progress/                    dated research history and audit evidence
outputs/                     ignored experiment artifacts and sync surface
ops/                         local workstation/agent policy helpers
.codex/skills/               tracked repo-local skills
```

Do not add a top-level `experiments/` until there is a concrete workflow that
is not already better handled by `configs/analysis`, `progress/`, and
`outputs/`. A generic `experiments/` folder would likely become another junk
drawer.

## Long-Term Governance Rules

1. New behavior starts in config, not hidden CLI flags.
2. New stable behavior must update docs and, when compatibility-sensitive,
   OpenSpec.
3. New experiment variants must declare lifecycle: `prod`, `smoke`,
   `ablation`, `negative`, `analysis`, or `bench`.
4. Analysis code can live in `src/analysis` only when tests import it or
   scripts reuse it. One-off notebooks or probes belong in `temp/`.
5. `scripts/analysis` wrappers should be thin: parse config, call
   `src/analysis`, write under `outputs/` or a configured output root.
6. `progress/` should keep conclusions, summaries, and exact artifact pointers,
   not whole copied run directories.
7. `outputs/` is the artifact store and sync surface. `output_remote/` is not a
   durable structure.
8. Raw and processed `public_data/` stay local; processed roots are reproduced
   from `manifests/public_data_provenance/`.
9. Local agent state stays local. Track skills, not memories, sessions, logs,
   auth, plugin caches, or app state.
10. Prefer deletion plus Git history over compatibility stubs after the
    operator route is updated.

## Cleanup Backlog

Safe, low-risk next actions:

- run `bash ops/workspace/workspace_gc.sh` dry-run, then delete Python caches,
  TensorBoard scratch, `result/`, and the stray root `(deleted)` after a quick
  glance;
- remove or move root credential/cookie files after confirming they are no
  longer needed by active sessions;
- prune tracked `progress/diagnostics/artifacts/` large files into summaries
  plus `outputs/` pointers;
- update `docs/catalog.yaml` to either include or intentionally exclude
  current `docs/training/*DRAFT*` and `docs/superpowers/*` surfaces;
- review dense config clusters after each experiment conclusion.

Needs human confirmation:

- deleting or moving `external/` checkouts;
- deleting model/cache/checkpoint roots;
- deleting raw or processed `public_data/` roots;
- any cleanup under `outputs/`;
- resolving whether `docs/superpowers/` should remain tracked design history
  or move under `progress/`.
