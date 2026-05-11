thread_id: 019de463-fe12-7d40-8ffd-99b8ce7cf93a
updated_at: 2026-05-07T02:48:41+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/05/01/rollout-2026-05-01T16-34-11-019de463-fe12-7d40-8ffd-99b8ce7cf93a.jsonl
cwd: /data/CoordExp
git_branch: main

# Exported the useful `rp=1.10` compact-full benchmark/union analysis to `progress/` and assessed Notion import suitability

Rollout context: the user asked to export all valuable results from the prior `val200` / `rp=1.10` analysis into local `progress/` and decide whether it is appropriate to import into Notion. The work centered on the top-3 compact-full rollout checkpoints, then the corrected bbox-only bootstrap-union estimate for unlabeled objects versus GT count.

## Task 1: Export the benchmark results and assess Notion suitability

Outcome: success

Preference signals:
- The user asked to "将之前的所有有价值的结果，导出到本地 `progress/`" -> future similar analysis results should be written into the repo-local historical layer, not left only in chat or temp artifacts.
- The user also asked to judge whether it is suitable to import into Notion -> future similar exports should include an explicit Notion recommendation, not just a repo note.
- After a clarification about deduplication, the user said the concrete union/dedup logic should use bbox overlap only -> future similar “union of objects” analyses should default to bbox-overlap identity, not class/description text identity.

Key steps:
- Retrieved the existing `progress/` router structure and confirmed the measured-result note belongs under `progress/benchmarks/` rather than `docs/` or `progress/diagnostics/`.
- Used the actual top-3 `rp=1.10` `val200` run artifacts to compute the corrected union relation.
- Iterated twice on the object-union definition until it matched the user’s intent: first avoided per-run FP summing, then changed from class/desc-aware dedup to bbox-overlap-only dedup, then applied union-vs-GT subtraction.
- Exported the final benchmark note and machine-readable JSON artifact into `progress/benchmarks/` and updated the router/index so the new result is discoverable.
- Added a Notion-import section that frames the result as research memory / claims-ledger material, not as a stable executable contract.

Failures and how to do differently:
- The first “unlabeled” estimate was a per-run median FP proxy; it was useful but ultimately superseded because it did not implement the user’s intended bootstrap union of unique objects across rollouts.
- The first union attempt used class/description as part of object identity; the user corrected this, and the final method should use bbox overlap only for both deduplication and GT subtraction.
- Temporary artifacts in `/data/CoordExp/temp/` were not sufficient by themselves; the durable deliverable needed a copy under `progress/benchmarks/artifacts/` plus a note in the benchmark router.

Reusable knowledge:
- In this repo, measured comparison / checkpoint-selection / sweep-style results belong under `progress/benchmarks/`, while `progress/diagnostics/` is better for root-cause or failure analysis.
- The `progress/` layer is historical/evidence, not current contract truth; the repo prefers keeping the benchmark note plus artifact links there, and only promoting to `docs/` when it becomes a stable current workflow.
- For this rollout’s scope, the authoritative object-union procedure became: union predictions across runs, deduplicate by bbox overlap only, then subtract GT by bbox overlap only.
- The final scoped prior derived from the top-3 `rp=1.10` compact-full `val200` runs was roughly `unlabeled_count ~= 0.38 to 0.40 * gt_annotation_count`, with a simpler through-origin prior of about `0.40 * G` and a heavy-burst-filtered linear fit of approximately `U ~= max(0, -0.35 + 0.43 * G)`.
- The final note explicitly marked the result as suitable for Notion only as research memory / claims ledger material, and explicitly warned against pasting the full per-image JSON into Notion.

References:
- [1] Created benchmark note: `progress/benchmarks/2026-05-07_compact_full_rp110_top3_union_unlabeled_prior.md`
- [2] Created compact artifact: `progress/benchmarks/artifacts/2026-05-07_compact_full_rp110_top3_union_summary.json`
- [3] Created per-image artifact: `progress/benchmarks/artifacts/2026-05-07_compact_full_rp110_top3_bbox_union_per_image.json`
- [4] Updated benchmark router: `progress/benchmarks/README.md`
- [5] Updated machine-readable index: `progress/index.yaml`
- [6] Final Notion guidance in the note: import as a `Research Unit` and `Claims Ledger` item only if the claim remains scoped to `val200`, top-3 `rp=1.10` compact-full rollouts, bbox-only IoU `0.50`, guarded prediction surface, and burst-filtered conditions.

## Task 1 (if needed): refine the union definition after user correction

Outcome: success

Preference signals:
- The user’s explicit example (`a,b` / `c,d` / `a,d` with GT `a`) showed they wanted set-union semantics over unique objects, not per-rollout counting.
- The later clarification "For the concrete `union`/`deduplication`, we shall just use the `bbox` overlapping" indicates the next-agent default should be bbox-overlap identity for object deduplication whenever similar multi-rollout bootstrap-union questions arise.

Reusable knowledge:
- If the user asks for “union of false positives/unlabeled objects,” the next agent should treat the analysis as a set problem: collect all candidate objects, deduplicate them by bbox overlap, then subtract GT matches.
- If collapse/burst cases appear, the note should preserve both a filtered primary estimate and an unfiltered sensitivity bound, because the unfiltered union can be dominated by a few pathological run-image cells.

References:
- Primary filtered estimate in the exported note: `U_union ~= max(0, -0.35 + 0.43 * G)` and `U_union ~= 0.40 * G`
- Sensitivity bound retained in the exported note: unfiltered union ratio about `0.4737` and `U_union ~= 0.46 * G`
- The per-image artifact and summary JSON were successfully parsed after export, and `progress/index.yaml` was verified in the `conda run -n ms` environment.
