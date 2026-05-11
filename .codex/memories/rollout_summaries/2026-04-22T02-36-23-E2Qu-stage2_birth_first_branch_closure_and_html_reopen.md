thread_id: 019db30b-b88b-77b3-8aee-bd3db9861e9e
updated_at: 2026-04-23T07:13:59+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/22/rollout-2026-04-22T02-36-23-019db30b-b88b-77b3-8aee-bd3db9861e9e.jsonl
cwd: /data/CoordExp
git_branch: main

# The user closed a Stage-2 birth-first research branch by preserving only durable docs/evidence on `main` and archiving the implementation branch tip.

Rollout context: cwd was `/data/CoordExp`. The research work happened in a transient worktree at `.worktrees/birth-first-stage2-channel-b`, but the user ultimately asked to close that branch and merge the important findings/docs into `main`. The session also contained a related debugging thread about a local HTML reviewer whose coordinates were mis-scaled because the overlay bundle was accidentally rendered on the source image size instead of the artifact-native 1000x1000 surface.

## Task 1: Stage-2 birth-first mechanism/audit synthesis and branch closure prep

Outcome: success

Preference signals:

- The user asked to treat their notes as "an initial qualitative audit, not as ground truth" and explicitly requested a falsifiable mechanism analysis that would "separate annotation noise from genuine model failure" and "update the explanation if the evidence contradicts my current intuition." This suggests that in similar research-audit situations, the next agent should actively challenge the user’s priors, separate latent causes, and frame conclusions as testable hypotheses rather than as endorsements of the user’s subjective read.
- The user later said, "I think it's time to close this research branch and merge the important findings and docs into the `main`." This indicates a durable preference for preserving only the stable findings/docs on `main`, not the experimental implementation path, once a research branch has matured.

Key steps:

- Read the birth-first decision study note and the prior duplication-collapse analysis to anchor the mechanism claims in repo-backed evidence.
- Used the local decision study to connect the user’s qualitative audit to concrete evidence: the birth-first enabled arm showed slightly higher recall but lower precision/F1 and much higher malformed/invalid outputs, with `dead_anchor_count = 0` and `neutral_shielded_count = 33` in enabled versus `26/0` in control.
- Used the duplication-collapse analysis to ground the core mechanism in a local coordinate basin / weak escape barrier story, especially around early `coord_x1` / `coord_y1` decisions, and to treat crowding as a trigger rather than a sufficient explanation.
- Preserved the stable artifacts by copying the small evidence bundle into `progress/diagnostics/artifacts/stage2_birth_first_channel_b_decision_study_2026-04-22/` and rewriting the decision note so it no longer pointed at transient `.worktrees/...` paths.
- Removed the transient superpower plan file from the merge set because it was worktree-specific and would become stale after closure.
- Merged the stable docs/specs and the cleaned-up progress note into `main` and archived the implementation tip with a tag before deleting the worktree and branch.

Failures and how to do differently:

- The first attempt to inspect local notes hit a sandbox/bwrap restriction; the pivot was to rerun the reads with escalated permissions.
- `openspec validate` was not available on the shell PATH, so validation had to be approximated with `git diff --cached --check` and direct path sanity checks.
- A stale `.git/index.lock` briefly blocked git operations; the fix was to remove the stale lock after confirming no live git process remained.
- The branch’s superpower execution plan was too worktree-specific to survive a merge to `main`, so it was intentionally excluded from the preservation commit.

Reusable knowledge:

- The most durable explanation for this family remains: a local geometry/coordination basin with weak early escape barriers, rather than generic hallucination or mere late attention overwrite.
- In the birth-first A/B decision study, the enabled arm was directionally alive but too permissive: it reduced dead anchors and raised recall slightly, but it also increased invalid outputs and lowered precision/F1.
- For branch closure in this repo, preserve the durable docs/specs and a compact evidence bundle under `progress/diagnostics/artifacts/`, then rewrite any note that still points at `.worktrees/...` before committing to `main`.
- If a transient research plan file is worktree-scoped, do not merge it into `main`; it will become stale as soon as the branch is closed.
- When a long-lived research branch is being retired, archive the branch tip with a tag before deleting the worktree/branch so the implementation snapshot remains recoverable.

References:

- [1] `progress/diagnostics/stage2_birth_first_channel_b_decision_study_2026-04-22.md` was rewritten to `status: consolidated-final` and repointed to stable paths under `progress/diagnostics/artifacts/stage2_birth_first_channel_b_decision_study_2026-04-22/`.
- [2] Stable evidence bundle copied into `progress/diagnostics/artifacts/stage2_birth_first_channel_b_decision_study_2026-04-22/` with `control_smallfrac_logging.jsonl`, `control_vllm_bonly_merged1332_logging.jsonl`, `enabled_smallfrac_logging.jsonl`, `control_smallfrac_launch.log`, and `enabled_smallfrac_launch.log`.
- [3] Merged commit on `main`: `9a76aa9` — `docs(stage2): preserve birth-first study findings`.
- [4] Archived implementation tip: `archive/birth-first-stage2-channel-b` pointing to branch commit `11af754` (`docs(stage2): fix birth-first adapter study contract`).
- [5] Branch/worktree cleanup: `.worktrees/birth-first-stage2-channel-b` was removed and `codex/birth-first-stage2-channel-b` was deleted.

## Task 2: Coordinate misalignment in local HTML reviewer and server reopen assistance

Outcome: success

Preference signals:

- The user repeatedly asked to “Help me re-open the html file so that I can web browser it,” and later clarified that the process was down and port `8766` was unavailable. This suggests that in similar browser-review situations, the next agent should be ready to reopen the local reviewer by restarting the serving process and giving a fresh cache-busted URL, rather than assuming the prior browser session or port remains valid.

Key steps:

- Verified the reviewer server and then traced a scaling bug to the review overlay using the source image dimensions instead of the artifact-native `1000 x 1000` render surface.
- Rewrote the reviewer bundle to preserve the real source image path but keep `width=1000`, `height=1000`, `coord_mode=pixel`.
- Re-generated the reviewer PNGs and added cache-busting query strings to the manifest URLs so the in-app browser would not keep showing stale content.
- Restarted the local HTTP server when the user reported port `8766` was down and then provided a fresh URL for reopening.

Failures and how to do differently:

- The first overlay fix was incomplete because the review bundle was re-pointed to the real source image but its render surface was also changed to the original image size; that caused coordinate drift even though the scene pairing itself was correct.
- The correct fix was to keep the artifact-native render surface and only remap the source image path.

Reusable knowledge:

- For these FP-review artifacts, the evaluation record is on a 1000x1000 render surface even when the underlying source image is non-square; overlays must respect the artifact-native surface.
- If the local reviewer server dies, `python -m http.server 8766 --directory /data/CoordExp` was sufficient to bring the HTML reviewer back, after which a cache-busted `index.html?v=3&ts=...` URL worked.

References:

- [1] `src/vis/gt_vs_pred.py:_coerce_bbox_from_object` shows the coordinate handling path that distinguishes `norm1000` versus `pixel` and denormalizes into the render surface.
- [2] The reviewer bundle was rewritten so scene entries kept `render_surface: {width: 1000, height: 1000, coord_mode: pixel}`.
- [3] The stable reviewer manifest was written to `/data/CoordExp/.worktrees/birth-first-stage2-channel-b/temp/fp_reviewer_ui/manifest.json` before the worktree was closed, and the HTML was reopened via `http://127.0.0.1:8766/.worktrees/birth-first-stage2-channel-b/temp/fp_reviewer_ui/index.html?v=3` when the server was live.
- [4] The reviewer server was later restarted with `python -m http.server 8766 --directory /data/CoordExp` after the user said port `8766` was down.

