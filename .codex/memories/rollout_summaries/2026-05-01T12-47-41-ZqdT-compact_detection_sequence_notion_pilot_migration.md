thread_id: 019de394-9eb9-7a63-ac73-281e62a26dcd
updated_at: 2026-05-01T13:13:28+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/05/01/rollout-2026-05-01T12-47-41-019de394-9eb9-7a63-ac73-281e62a26dcd.jsonl
cwd: /data/CoordExp
git_branch: main

# Implemented a simplified Notion migration pilot for the `codex/compact-detection-sequence` worktree and updated the repo-side pilot docs to match.

Rollout context: the user wanted to switch to the existing `codex/compact-detection-sequence` worktree, initialize Notion, and show how to migrate `progress/` into a Notion-style management surface. They then refined the request to make `CoordExp` the global project root in Notion and to rename/optimize `progress` into a project-oriented evidence area. The rollout happened in `/data/CoordExp` and then in the worktree at `/data/CoordExp/.worktrees/compact-detection-sequence`.

## Task 1: Plan the Notion migration and worktree setup

Outcome: success

Preference signals:
- The user repeatedly asked to “checkout/switch to `codex/compact-detection-sequence` worktree and initialize the `Notion`” and then said “Please further simplify the `rules/split` for the first experimental attempt.” -> future similar requests should default to a small first-pass setup, not a full management system.
- The user explicitly narrowed the first attempt to “one Notion page” and later to a global `CoordExp` root with `progress` migrated into it as “renamed/optimized progress/experiment recordings.” -> future agents should assume the user prefers a compact pilot surface first, with later structure only if it proves valuable.
- The user asked for `CoordExp` as the codebase/project root in Notion. -> future migrations should treat `CoordExp` as the top-level Notion project home when possible.

Key steps:
- Confirmed the requested branch already existed as `/data/CoordExp/.worktrees/compact-detection-sequence`.
- Inspected the existing Notion and CoordExp workflow skills and the repo’s `progress/index.yaml` and `docs/AGENT_INDEX.md` / `docs/catalog.yaml` routing.
- Verified there was no existing CoordExp Notion page or teamspace in the workspace search.
- Determined the first-pass scope should be one Notion project root plus one experiment page, not a full database schema.

Reusable knowledge:
- In CoordExp, `docs/` remains current/stable truth and `progress/` remains historical evidence; this was reinforced by both repo docs and the user’s request to migrate `progress` into a Notion-style evidence surface.
- The Notion connector in this workspace would not accept a top-level page directly without a parent page, but a page could be created under an existing page and then moved to workspace level.
- Notion page content is safer when compact sequence markers are put in code blocks rather than table cells, because raw `<|desc|>`-style tokens can be misparsed as markup.

Failures and how to do differently:
- The first attempt to create a workspace-level page failed because the connector schema required a parent page. The workaround was to create under an existing page (`Welcome to Notion`) and then move it to workspace level.
- The first version of the pilot page used a table containing raw compact-sequence markers and Notion mangled those strings. The fix was to rewrite those sequence sketches as code blocks.

References:
- Existing worktree: `/data/CoordExp/.worktrees/compact-detection-sequence`
- Notion roots created: `CoordExp`, `Experiments & Evidence`, `Compact Detection Sequence Pilot`
- Workspace fetch confirmation showed the pages were present and the content was corrected after the code-block rewrite.

## Task 2: Create the Notion project root and migration pages

Outcome: success

Preference signals:
- The user asked for a global `CoordExp` root and for `progress` to be migrated under that project with renamed/optimized recordings. -> future similar setups should create a project root first, then migration pages underneath it.
- The user specifically asked for “renamed/optimized progress/experiment recordings.” -> future agents should not mirror repository labels literally when the user asks for a more management-friendly surface.

Key steps:
- Created a workspace-level Notion page `CoordExp`.
- Created a child page `Experiments & Evidence` and then moved it to workspace level so it acts as the migration root.
- Created `Compact Detection Sequence Pilot` as the experiment control page.
- Added renamed category pages under `Experiments & Evidence`:
  - `Research Directions`
  - `Mechanism & Failure Records`
  - `Result Records`
  - `Audit Records`
  - `Architecture Explorations`
  - `Stage-1 Foundation`
- Used `progress/index.yaml` as the source map for the renamed Notion categories and curated records.

Reusable knowledge:
- A single Notion “project root” can work as the entry point, with a second “evidence” page acting as the migration surface and a third page acting as the experiment control room.
- For this workspace, the global/workspace-level Notion page can be obtained by create-under-parent followed by move-to-workspace.

Failures and how to do differently:
- The first create-page attempts with an omitted parent or a direct workspace parent failed schema validation. The success path was: create under existing parent page, then move the created page to workspace.

References:
- `https://app.notion.com/p/3539d9ce3f59814fad41ce04ae1e42a9` — `CoordExp`
- `https://app.notion.com/p/3539d9ce3f59813dbff8f439549b92cc` — `Experiments & Evidence`
- `https://app.notion.com/p/3539d9ce3f5981bba7acfc34ea12441a` — `Compact Detection Sequence Pilot`
- Renamed category pages created under the migration root and verified by fetch.

## Task 3: Simplify the repo-side pilot docs

Outcome: success

Preference signals:
- The user explicitly asked to “simplify the `rules/split` for the first experimental attempt,” then later reiterated a stronger simplification: `CoordExp` as project root, `progress` as curated experiment/evidence recordings. -> future docs should start with the simple split and only add structure when the experiment proves it needs more.
- The user wanted the first attempt to avoid overbuilding, so the assistant should default to a lightweight pilot page and a minimal workflow split.

Key steps:
- Updated `docs/superpowers/research-management-pilot.md` to describe the `CoordExp` Notion root, the `Experiments & Evidence` migration surface, and the `Compact Detection Sequence Pilot` page.
- Updated `docs/superpowers/specs/2026-05-01-compact-detection-sequence-ablation-design.md` to say the first attempt uses `CoordExp / Experiments & Evidence / Compact Detection Sequence Pilot`, skips Linear, and keeps measured results in `progress/benchmarks/`.
- Updated `docs/superpowers/plans/2026-05-01-compact-detection-sequence-ablation.md` to replace the old Notion+Linear task graph with a one-page Notion pilot and to require the final `progress/benchmarks/2026-05-01_compact_detection_sequence_val200.md` note after measured results exist.

Reusable knowledge:
- The repo-local pilot docs now encode a simplified first-pass governance model:
  - repo = executable truth
  - Notion = readable control room + migration surface
  - `progress/` = final evidence only
  - `docs/` = promoted stable behavior
  - OpenSpec = only if stable contracts change
  - Linear = skipped for the first attempt unless coordination becomes painful
- The Notion migration note is now framed as a compact evidence-card format rather than a full Markdown dump.

Failures and how to do differently:
- The original plan text was too ceremony-heavy for the user’s first experiment. The revision removed the old Notion+Linear task graph and replaced it with the smaller Notion-first workflow.
- The plan file remains untracked in the worktree; future agents should remember that `git diff -- docs/superpowers` can be empty when those docs are newly untracked, even though the files exist and were edited.

References:
- `docs/superpowers/research-management-pilot.md`
- `docs/superpowers/specs/2026-05-01-compact-detection-sequence-ablation-design.md`
- `docs/superpowers/plans/2026-05-01-compact-detection-sequence-ablation.md`
- Notion pilot page content now includes the compact evidence-card schema and renamed category maps.

