thread_id: 019db06f-a7ec-7c41-b1cd-f6779b247bcd
updated_at: 2026-04-23T06:47:16+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/21/rollout-2026-04-21T14-26-40-019db06f-a7ec-7c41-b1cd-f6779b247bcd.jsonl
cwd: /data/CoordExp
git_branch: main

# Raw-text decode-bias study was completed in an isolated worktree, then merged back to main after resolving progress-router conflicts and separately committing an unrelated main-line progress note.

Rollout context: The user asked for a research task about raw-text decode-time bias (EOS/continue length bias and repeat-penalty bias under dense same-class enumeration), then later asked to use a worktree, create the spec there, and eventually merge the worktree back while preserving what was learned so far. The work ultimately produced a new decode-bias study stack, a spec and plan, a preserved diagnostic note, and a merged `main` state with verification passing.

## Task 1: Scope the raw-text decode-bias research and choose a study design

Outcome: success

Preference signals:
- The user explicitly asked for a research-scale direction and later answered the scope questions with “Both! Good question for the research-scale direction.” -> this indicates they wanted both teacher-forced counterfactual scoring and fresh HF decode sweeps, not a single-lane analysis.
- When asked for the benchmark surface, the user chose “val200” -> this suggests future similar study work should default to `val200` when a broad validation surface is needed.
- When asked about the EOS leg, the user chose the targeted stop-pressure ablation option -> this indicates a preference for narrow, interpretable intervention rather than a broad decode-policy search.
- When asked about repeat penalty, the user accepted the suggested small grid -> this indicates tolerance for a compact sweep around the current default instead of a large parameter search.

Reusable knowledge:
- The repo already had a raw-text mechanism study stack plus dedicated probe scripts; the right move was to extend that stack rather than create a parallel harness.
- Existing code paths already supported teacher-forced matched-span scoring, repetition-penalty sweeps, and stop-pressure-style decode experiments, so the study could reuse those seams.
- The eventual final mechanistic conclusion preserved in progress notes was that the harmful decision point is the abstract stop-vs-continue branch at the end of an object, not a literal EOS token identity.

References:
- Existing raw-text mechanism study stack: `configs/analysis/raw_text_coordinate_mechanism/`, `scripts/analysis/run_raw_text_coordinate_mechanism_study.py`, `src/analysis/raw_text_coordinate_mechanism_study.py`
- Reused scoring seams: `src/analysis/raw_text_coordinate_continuation_scoring.py`, `src/analysis/raw_text_coord_continuity_scoring.py`, `src/analysis/unmatched_proposal_verifier.py`
- Final preserved note: `progress/diagnostics/raw_text_decode_bias_mechanism_findings_2026-04-22.md`

## Task 2: Draft spec / plan in a worktree and validate the existing mechanism-study surface

Outcome: success

Preference signals:
- When the user said “use the worktree skill to keep separation clean” and asked to move to a worktree for spec creation and audit, that indicates they prefer isolated research branches / worktrees for multi-step research work.
- The user later said “Good, I think it's enough for this research track/worktree.” and “Let's merge this worktree and make sure we record down what we've got so far.” -> this indicates they wanted the worktree preserved as a research track until the evidence was recorded, then merged.
- When they later asked to “help me manage to commit on the main and then merge this worktree,” they were explicitly asking for separate main-line and research-branch integration rather than an ad hoc direct merge.

Reusable knowledge:
- `.worktrees/` exists and is ignored in this repo; the worktree setup path used a dedicated branch under `.worktrees/agent-raw-text-decode-bias`.
- The worktree baseline tests passed cleanly before editing, which was useful for proving the workspace started healthy.
- The merge path needed a clean integration worktree because the main checkout had unrelated dirty `progress/` edits overlapping the same router files.

Failures and how to do differently:
- A direct merge into the dirty `/data/CoordExp` checkout was unsafe because it would have mixed the user’s unrelated progress edits with the decode-bias merge.
- The safe pattern that worked was: commit the research branch, create a clean merge worktree, resolve the router-file conflicts there, verify on the merged tree, and only then land on `main`.

References:
- Worktree path: `/data/CoordExp/.worktrees/agent-raw-text-decode-bias`
- Clean integration worktree: `/data/CoordExp/.worktrees/merge-raw-text-decode-bias-main`
- Baseline verification in worktree: focused pytest subset passed (`7 passed in 0.86s`)
- Later focused verification on merged trees: `85 passed` and `yaml_ok`

## Task 3: Implement, verify, and record the raw-text decode-bias study

Outcome: success

Preference signals:
- The user accepted the proposed split between counterfactual scoring and end-to-end sweeps, which indicates they wanted both causal isolation and operational impact measurement.
- The user later redirected the discussion away from literal EOS tokens and toward the abstract stop decision -> this suggests future similar work should frame the intervention in terms of branch classes (`stop_now`, `continue_with_next_object`, `wrong_schema_continuation`) rather than surface token identity.
- The user asked whether loosening EOS pressure would predict more correct objects, then whether the adapter showed any improvement when forced not to stop -> this suggests the user cares about concrete negative/positive evidence, not just conceptual explanation.

Reusable knowledge:
- The decode-bias study’s strongest preserved conclusion is that the apparent “EOS” issue is really an abstract premature-list-termination / stop-vs-continue branchpoint problem.
- The final note records that special EOS suppression was inert, blunt structural suppression was harmful, persistent continuation steering caused runaway enumeration, and one-shot post-closure rescue was an exact no-op.
- The broader `EOS-hard12` rerun produced exact `off` vs `on` equality for both checkpoints on `pred`, `raw_output_json`, `errors`, and `generated_token_text`, which makes the no-op conclusion strong, not just metric-flat.

Failures and how to do differently:
- Early “don’t stop” interventions should not be interpreted as a generic fix for missed objects; the evidence shows they often either do nothing or make output quality worse.
- For the adapter checkpoint specifically, “force not stop” did not improve results; repeat penalty was the decode-time lever that showed real benefit.
- The final research track should be framed as “increase valid next-object pressure” rather than “suppress EOS tokens.”

References:
- Final study note: `progress/diagnostics/raw_text_decode_bias_mechanism_findings_2026-04-22.md`
- Branchpoint census / EOS-hard12 rerun artifacts:
  - `/data/CoordExp/output/analysis/raw-text-decode-bias-base-only-stop-signature19-branchpoint-census`
  - `/data/CoordExp/output/analysis/raw-text-decode-bias-eos-hard12-bbox-tail-then-object-open-once-bs4-rerun`
- Key preserved data points from the note:
  - wrong-schema top token `"],"` in `19/19`
  - close-now top token `"]}"` in `18/19`
  - next-object top token almost absent (`" ,"` dominates a tiny mass)
  - `base_only` and `base_plus_adapter` both exact-no-op under the broader one-shot stop-pressure rerun

## Task 4: Commit main-line progress note, merge the worktree, and clean up

Outcome: success

Preference signals:
- The user explicitly asked to “commit on the main and then merge this worktree” -> that indicates they want main-line notes committed separately before integrating research work.
- After seeing the separate main-line progress note, the user accepted keeping the verified merge prepared safely and then later asked to proceed with merging -> this suggests the user prefers the safe, staged integration flow over forcing a messy direct merge.
- The user accepted the “keep the verified merge on the integration branch for now, and fast-forward main once clean” option -> that reinforces the preference for safe integration over forcing in-place changes on a dirty checkout.

Reusable knowledge:
- The main checkout had unrelated dirty `progress/` edits that overlapped the router files used by the decode-bias worktree. That made direct merge into `/data/CoordExp` unsafe until the main-line progress note was committed separately.
- The safe resolution path was:
  1. commit the unrelated main-line progress note on `main`,
  2. replay that commit into the clean integration worktree,
  3. merge the verified decode-bias branch on top,
  4. verify on the merged `main`,
  5. remove the temporary worktrees and delete the local branches.
- The final merged `main` passed the focused decode-bias pytest subset (`85 passed in 1.23s`) and parsed `progress/index.yaml` successfully (`yaml_ok`), and a diff-based cleanliness check reported `clean`.

Failures and how to do differently:
- A merge into the clean integration worktree initially conflicted on `progress/diagnostics/README.md` and `progress/index.yaml`; the conflict was resolved by preserving the main-line progress-router structure and inserting the new decode-bias note as a new active reference under the same cluster.
- The worktree registry needed pruning after removal; `git worktree prune` was required to fully clear the stale entries.
- `git status` output was oddly empty in this shell at first; a direct `git diff --quiet && git diff --cached --quiet` check was the reliable final cleanliness proof.

References:
- Main commit for the unrelated progress note: `a72d8eb` (`progress: add birth-first channel-b decision note`)
- Final merged main commit: `ec59d7f` (`merge: integrate raw-text decode-bias worktree`)
- Verified integration branch commit before merge: `05d0491` / `4a90ec9` lineage in the integration worktree
- Final merged `main` status check: `clean`
- Deleted worktrees:
  - `/data/CoordExp/.worktrees/agent-raw-text-decode-bias`
  - `/data/CoordExp/.worktrees/merge-raw-text-decode-bias-main`
- Deleted local branches:
  - `agent-raw-text-decode-bias`
  - `codex/merge-raw-text-decode-bias-main`
