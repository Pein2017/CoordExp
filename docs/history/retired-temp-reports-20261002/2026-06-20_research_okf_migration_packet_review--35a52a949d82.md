# CoordExp OKF Research Migration Packet Review

Mode: read-only critical review, finalized as a local Markdown deliverable at the user's request.

Reviewed worktree: `/data/CoordExp/.worktrees/progress-okf-union-collection`

Primary artifacts:

- `docs/superpowers/specs/2026-06-20-research-okf-migration-design.md`
- `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md`
- `progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md`

Supporting artifacts:

- `docs/catalog.yaml`
- `progress/index.yaml`
- `docs/AGENT_INDEX.md`
- `docs/history/README.md`
- `progress/explorations/README.md`
- `docs/history/worktree-union/2026-06-20/manifest.tsv`
- Required prefix-denoising source notes listed by the design and plan

Upstream reference checked: [GoogleCloudPlatform knowledge-catalog OKF SPEC.md](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md). I used it only to sanity-check the local packet's OKF-style assumptions, not as a normative dependency.

## Findings

### P1 - Provenance validation is too weak to prove the manifest facts survive migration

Evidence:

- The design makes provenance preservation a core objective and names the raw worktree-union manifest as source truth: `docs/superpowers/specs/2026-06-20-research-okf-migration-design.md:9`, `:17-18`, `:211-214`.
- The implementation plan correctly says the source map should preserve branch, worktree, commit, snapshot hash, same-path status, dirty/untracked status, and snapshot paths from the manifest: `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md:137-140`.
- The executable validation only checks that `overview.md` contains several source path strings plus the two hash prefixes `688fa6f04eb5` and `ed610b284219`: `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md:780-794`.
- The manifest has richer and risk-bearing prefix-denoising rows that the current validation would not require the pilot to preserve, including branch-head vs dirty source kind and untracked status for the root-cause note: `docs/history/worktree-union/2026-06-20/manifest.tsv:64-65`. It also has relevant non-normative OpenSpec/config/docs snapshot rows on the same branch that should remain provenance, not current contract: `docs/history/worktree-union/2026-06-20/manifest.tsv:63`, `:107-119`.

Why it matters:

The pilot could pass the planned validator while losing exactly the provenance that makes the side-by-side migration safe: whether a note came from branch head or dirty worktree state, whether the inert-objective analysis was untracked, which snapshot path captured it, and which branch/OpenSpec material should remain historical rather than normative. That is a reproducibility and authority-drift risk, not just a documentation nicety.

Recommended fix:

Before approving implementation, strengthen Task 4 and Task 8 so at least `overview.md` or `implementation.md` contains a compact "Manifest provenance" table for the required prefix-denoising rows. Include: manifest line or row id, `classification`, `source_kind`, `branch`, `head`, `worktree`, `source_path`, `status`, `sha256`, `same_path_in_main`, `content_paths_in_main`, and `snapshot_path`. For the OpenSpec rows, explicitly label them as historical branch snapshots unless a later stable contract change is separately approved.

Verification suggestion:

Add a small validation step that parses `docs/history/worktree-union/2026-06-20/manifest.tsv`, filters `branch == "codex/prefix-denoising-sft"` plus required source paths, and asserts that each required row's `sha256`, `source_kind`, `status`, and `snapshot_path` appear in the generated pilot. This should be an artifact check, not a broad git cleanup.

### P1 - The no-match `rg` hygiene checks are self-failing in common execution contexts

Evidence:

- The plan asks implementers to run two `rg` commands that are expected to find no matches, then states that each should exit `1` with no output: `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md:825-840`.
- The same plan tells agentic workers to execute task-by-task with a superpower execution workflow: `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md:3`.

Why it matters:

Plain `rg` uses exit code `1` for "no matches." That is semantically success here, but many task runners, shell snippets with `set -e`, or cautious implementation agents will treat the first `rg` as a failed validation command and stop or misreport the migration as broken. The packet explicitly asks for scoped, runnable validation that is not self-failing, so this should be fixed before approval.

Recommended fix:

Replace the raw no-match `rg` commands with inverted checks that exit `0` on no matches and `1` on residue. For example:

```bash
if rg -n "projects/|concepts/|type: project|type: note|type: experiment|type: concept|programs/|threads/|lines/|new_idea" research docs/AGENT_INDEX.md docs/catalog.yaml progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md; then
  echo "unexpected rejected naming residue" >&2
  exit 1
fi

if rg -n "[ \t]$" research docs/AGENT_INDEX.md docs/catalog.yaml progress/index.yaml progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md; then
  echo "unexpected trailing whitespace" >&2
  exit 1
fi

git diff --check -- research docs/AGENT_INDEX.md docs/catalog.yaml progress/index.yaml progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md
```

Verification suggestion:

Run the replacement block after implementation and report all three exit codes. Expected: the two inverted `rg` checks exit `0` with no matches; `git diff --check` exits `0`.

### P1 - Source-link requirements and link validation do not prove per-note source grounding

Evidence:

- The plan requires `overview.md` to "Link back to every required source listed in Task 1": `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md:341-344`.
- Experiment notes are only required to synthesize from the relevant source note, but the plan does not require each experiment note to carry its own source/provenance handle beyond a generic `## Source` or `## Sources` section: `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md:471-575`.
- The validator checks only `overview.md` for source-handle substrings and hash prefixes, not the three experiment notes, `draft.md`, `discussion.md`, or `implementation.md`: `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md:780-794`.
- The generic Markdown link checker resolves every local link relative to the current research file: `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md:796-805`. That is fine for internal `research/` links, but it does not define a clear convention for repo-root provenance links to `progress/` and `docs/history/`.

Why it matters:

The synthesized reading path could become a polished but weakly sourced summary. In particular, the axis-sort and inert-objective experiment notes need local source handles because their evidence has different status and scope: the axis-sort source is a branch-head snapshot; the inert root-cause source is a dirty-worktree untracked snapshot. If only `overview.md` contains source handles, future readers may quote the experiment note without seeing its provenance caveat.

Recommended fix:

Require every non-router pilot document to end with a `## Sources` section that lists the exact source paths and, when applicable, manifest row/hash/snapshot handles. Then split validation into two checks:

- internal research links must resolve relative to `research/`;
- provenance handles to `progress/`, `docs/history/`, artifacts, commits, and worktrees may be repo-root path handles or correctly relative Markdown links, and must be present in the document that uses the claim.

Verification suggestion:

Add a source-coverage check mapping each generated file to its expected source paths. For example, the axis-sort note must include the `688fa6f04eb5` snapshot path and the original inference artifact; the inert-objective note must include `ed610b284219`, the dirty/untracked provenance status, and the matched-control gap.

### P2 - `research/index.md` wording slightly blurs provenance ownership

Evidence:

- The planned root `research/index.md` says to use `research/` for "research development, idea lifecycle records, investigations, mechanism notes, experiment interpretation, negative results, and provenance": `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md:159-164`.
- The design and alignment note separately state that raw branch/worktree intake remains non-normative under `docs/history/`: `docs/superpowers/specs/2026-06-20-research-okf-migration-design.md:223-228`; `progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md:40-41`, `:256-257`.
- `docs/history/README.md` already defines `docs/history/` as historical, non-current provenance: `docs/history/README.md:14-23`.

Why it matters:

This is not a blocking authority drift because the nearby text correctly preserves `docs/` and `openspec/`. Still, "and provenance" is broad enough that a future worker might move raw worktree intake into `research/archive/` rather than linking to `docs/history/worktree-union/`.

Recommended fix:

Change the root router wording to "provenance handles" or "source-linked interpretation" and explicitly say raw worktree intake remains under `docs/history/worktree-union/`.

Verification suggestion:

After implementation, search the generated `research/` tree for large copied raw snapshot bodies or raw `docs/history/worktree-union/` content. Expected: only links/handles, no copied raw intake.

### P2 - The planned `docs/AGENT_INDEX.md` router snippet should include a current-behavior warning for `research/`

Evidence:

- The current `docs/AGENT_INDEX.md` has a crisp `progress/` usage rule: use it only for historical or empirical questions, and do not answer current behavior from `progress/` when `docs/` or `openspec/specs/` cover it: `docs/AGENT_INDEX.md:92-101`.
- The planned new research snippet says `progress/` remains historical/evidence source of truth and `research/` is a synthesized reading path, but it does not include the same "do not answer current behavior from this surface" warning: `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md:590-596`.
- The alignment note does include the stronger rule: do not use `research/` to answer current coding, architecture, infrastructure, or operator behavior when `docs/` or `openspec/` define it: `progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md:313-317`.

Why it matters:

The router is the place future agents are most likely to read first. If the stronger rule stays only in the alignment note, agents may over-read the new `research/` pilot as current behavior truth.

Recommended fix:

Add one line to the planned `docs/AGENT_INDEX.md` snippet:

```markdown
  - Do not answer current coding, architecture, infrastructure, operator, schema, artifact, metric, or training/eval behavior from `research/` when `docs/` or `openspec/specs/` cover it.
```

Verification suggestion:

After implementation, check `docs/AGENT_INDEX.md` contains both the `research/` entrypoint and the current-behavior warning.

### P2 - The local OKF validator is intentionally stricter than upstream; say that explicitly

Evidence:

- The design intentionally disallows `okf_version` frontmatter in `research/index.md` in phase 1 and keeps every `index.md` frontmatter-free: `docs/superpowers/specs/2026-06-20-research-okf-migration-design.md:41-44`.
- The validation enforces no frontmatter on every `index.md` and `log.md`: `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md:647-688`.
- Upstream OKF v0.1 treats `index.md` and `log.md` as reserved, requires non-reserved Markdown files to have parseable frontmatter with non-empty `type`, and allows permissive consumption. It also permits a root `index.md` version declaration as an exception, which CoordExp is choosing not to use in phase 1.

Why it matters:

This is not wrong. It is a reasonable local simplification. The small risk is future reviewer confusion: someone may read the upstream exception and think the CoordExp validator is accidentally non-conformant.

Recommended fix:

Add one sentence to the validation section: "Phase 1 deliberately uses a stricter local convention than upstream OKF by keeping all `index.md` files frontmatter-free and omitting `okf_version`."

Verification suggestion:

No runtime check needed. Verify the sentence is present near the Task 8 frontmatter validator.

## Confirmed OK Checks

- No P0 findings. I did not find an internal contradiction that makes the roadmap unsafe before any implementation.
- The authority model is mostly well preserved. The design keeps current behavior in `docs/`, stable compatibility-sensitive contracts in `openspec/specs/`, research interpretation in `research/`, and raw intake under `docs/history/`: `docs/superpowers/specs/2026-06-20-research-okf-migration-design.md:221-232`.
- The first implementation phase is correctly scoped as a side-by-side pilot under `research/ideas/prefix-denoising-sft/`, with no `progress/` rename or deletion: `docs/superpowers/specs/2026-06-20-research-okf-migration-design.md:27-37`; `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md:20-31`, `:877-881`.
- The packet correctly avoids adding an OpenSpec change for this documentation/knowledge-base migration unless stable contracts or operator workflows change: `docs/superpowers/specs/2026-06-20-research-okf-migration-design.md:45-46`; `progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md:307-309`.
- The type vocabulary is minimal and consistent with the intended local OKF use: `idea`, `investigation`, `mechanism`: `docs/superpowers/specs/2026-06-20-research-okf-migration-design.md:72-77`; `progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md:113-124`.
- The hierarchy `ideas/`, `investigations/`, `mechanisms/`, `archive/` is sufficient for the pilot. I found no leakage of rejected target names into the target tree, only into rejected-name checks: `docs/superpowers/specs/2026-06-20-research-okf-migration-design.md:79-149`, `:245-248`.
- The pilot correctly treats `prefix-denoising-sft` as an active idea with incomplete lifecycle and no `conclusion.md`: `docs/superpowers/specs/2026-06-20-research-okf-migration-design.md:151-172`; `docs/superpowers/plans/2026-06-20-research-okf-migration-plan.md:255`, `:334-340`.
- The planned scientific interpretation is appropriately cautious. Source evidence supports "launch-health repaired," "axis-sort improves materialization but not localization," and "inert objective plus non-comparable baseline, not proof against geometry-aware denoising": `progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md:5-11`; `progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md:11-27`, `:51-62`; `docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md:68-91`; `docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md:14-18`, `:73-89`, `:151-153`.
- Required source files named by the plan exist in the reviewed worktree. This was checked directly with a `Path.is_file()` probe.
- `docs/catalog.yaml` currently has the nested `entrypoints.human` and `entrypoints.agent` shape expected by the plan, and its authority keys are still the existing `docs/`, `docs/history/`, `openspec/specs/`, `openspec/changes/<active-change>/`, and `progress/` entries: `docs/catalog.yaml:4-18`.

## Open Questions For The User

- Should the first pilot preserve only the manifest rows directly used by the prefix-denoising reading path, or all `codex/prefix-denoising-sft` rows from the union manifest as a compact provenance appendix?
- Should the future full migration actually rename `progress/` to `research/`, or should `progress/` remain indefinitely as a historical evidence pool while `research/` becomes the curated semantic layer? The packet is clear that phase 1 is side-by-side, but the final rename is still a user-level governance decision.
- Should branch OpenSpec snapshots from the prefix-denoising worktree be listed in the pilot only as historical provenance, or should they be explicitly excluded from the idea reading path to avoid any contract-authority confusion?

## Final Recommendation

Approve after P1 fixes.

The roadmap direction is sound and the pilot scope is appropriately narrow. Do not approve implementation until the plan fixes provenance preservation, per-note source grounding, and self-failing validation commands. The P2s can be fixed in the same small doc update or acknowledged and deferred, but they should not block once the P1 items are patched.

## Suggested Next Actions

1. Patch the implementation plan before starting migration:
   - add manifest-row provenance requirements and validation;
   - make per-note source sections mandatory;
   - replace raw no-match `rg` checks with inverted checks.
2. Add the `research/` current-behavior warning to the planned `docs/AGENT_INDEX.md` router text.
3. Then approve the side-by-side pilot only. Do not approve mass migration, `progress/` rename, new `research/investigations/` content, or OpenSpec changes in the same pass.
