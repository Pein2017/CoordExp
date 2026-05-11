thread_id: 019d71bb-b19c-70c1-a226-71dd60d80c31
updated_at: 2026-04-13T02:22:16+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/09/rollout-2026-04-09T10-13-39-019d71bb-b19c-70c1-a226-71dd60d80c31.jsonl
cwd: /data/CoordExp
git_branch: main

# Consolidated the duplication-collapse investigation into one final progress document and removed superseded study notes/scratch artifacts.

Rollout context: The user asked for a final cleanup, proper runtime-artifact organization, and a single comprehensive analysis document under `docs/progress`. The assistant re-read the study markdowns and supporting Stage-1 docs/benchmarks, then consolidated the conclusions into one durable report and cleaned up obsolete intermediate notes.

## Task 1: Re-read and consolidate duplication-collapse findings

Outcome: success

Preference signals:
- The user asked: "Please perform a final cleanup and properly organize all runtime artifacts. Then merge all documents and findings into a single comprehensive analysis document under `docs/progress`." -> future similar requests should default to producing one standalone canonical document instead of leaving a constellation of study notes.
- The user required: "Re-read all `*.md` files before merging to ensure consistency and completeness." -> future similar consolidations should explicitly re-open the source markdown before synthesis, not rely on memory or prior summaries.
- The user required: "Produce a thorough, well-structured final document." -> future similar consolidation docs should be comprehensive, sectioned, and self-contained.
- The user required: "Remove all intermediate or individual documents after consolidation." -> after merging, superseded study-specific markdown should be deleted rather than retained alongside the final doc.

Key steps:
- Re-read the study proposal/design/spec/tasks under `openspec/changes/add-duplication-collapse-analysis-study/` before removing them.
- Re-read the follow-up research note and executive task list under `research/duplication_followup/`.
- Re-read the supporting Stage-1 / data / benchmark docs that informed the conclusions: `docs/training/STAGE1_OBJECTIVE.md`, `docs/data/CONTRACT.md`, `progress/pretrain/stage1_ablation_2026-01-26.md`, `progress/benchmarks/stage1_training_dynamics_4b_2026-02-26.md`, and `progress/benchmarks/stage1_coco80_4b_res_768_vs_1024_2026-02-26.md`.
- Wrote one consolidated final analysis document at `docs/progress/duplication_collapse_final_analysis_2026-04-13.md` with:
  - the study scope and runtime-artifact map,
  - the fixed-decode / fixed-checkpoint method,
  - the checkpoint-family interpretation,
  - the converged mechanism findings,
  - retraining and parameterization implications,
  - and the remaining open questions.
- Explicitly preserved the stable runtime artifact roots in the final document instead of moving them, to avoid breaking manifests and downstream readers.

Failures and how to do differently:
- The user wanted the final record consolidated under `docs/progress`, but the existing study notes lived in `openspec/changes/...` and `research/...`; the assistant initially had to inspect both trees before writing the final doc. In similar cleanup tasks, go directly to a single canonical final location and treat the rest as source material to be retired.
- A temporary scratch config under `temp/` remained after the first cleanup pass; it was removed in a final sweep. In similar cases, always do a second pass over `temp/` for lingering study-specific scratch files.

Reusable knowledge:
- The final durable analysis artifact is `docs/progress/duplication_collapse_final_analysis_2026-04-13.md`.
- The consolidation preserved stable research roots rather than relocating them; the final document itself now serves as the index to the surviving runtime artifacts.
- The conclusions recorded in the final doc are the current best-supported synthesis: a local coordinate basin / weak early escape barrier centered on `coord_x1` and `coord_y1`, with `predicted_object` vs `exact_duplicate` as the preferred causal probe, late history-overwrite as a secondary amplifier, crowding as a strong trigger but not sufficient, and pure CE as the best current mechanistic baseline without being immune.

References:
- [1] Final consolidated analysis: `docs/progress/duplication_collapse_final_analysis_2026-04-13.md`
- [2] Source study notes that were re-read before consolidation: `openspec/changes/add-duplication-collapse-analysis-study/{proposal.md,design.md,specs/duplication-collapse-analysis-study/spec.md,tasks.md}` and `research/duplication_followup/{executive_task_list.md,findings_2026-04-13.md}` (later removed after merge)
- [3] Supporting docs re-read for consistency: `docs/training/STAGE1_OBJECTIVE.md`, `docs/data/CONTRACT.md`, `progress/pretrain/stage1_ablation_2026-01-26.md`, `progress/benchmarks/stage1_training_dynamics_4b_2026-02-26.md`, `progress/benchmarks/stage1_coco80_4b_res_768_vs_1024_2026-02-26.md`
- [4] Cleanup verification: superseded markdown under the old study roots was removed; remaining duplication-study scratch files in `temp/` were deleted after the final sweep

