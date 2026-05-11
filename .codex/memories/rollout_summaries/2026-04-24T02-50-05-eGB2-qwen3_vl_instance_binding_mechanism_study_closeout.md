thread_id: 019dbd64-fcd3-71c0-a947-e2b090a65370
updated_at: 2026-04-24T08:03:02+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/24/rollout-2026-04-24T02-50-05-019dbd64-fcd3-71c0-a947-e2b090a65370.jsonl
cwd: /data/CoordExp
git_branch: main

# Fixed-checkpoint Qwen3-VL instance-binding mechanism study was designed, executed in a worktree, promoted to `main`, and then closed with the progress note canonicalized.

Rollout context: the user asked for a mechanism-first study on a fixed CoordExp / Qwen3-VL coord-token checkpoint (`output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full`) to determine whether same-desc instance binding is already present before `x1` or whether the decisive split still happens at `x1/y1`. They explicitly wanted a plan/spec pass first, then later asked to promote the progress/conclusion into `main` and close the worktree.

## Task 1: Mechanism-first study design and execution

Outcome: success

Preference signals:

- the user repeatedly said “Plan and brainstorm only. Don't execute now. Draft the relevant super-power docs first and will implement in the worktree.” -> in similar research requests, the agent should default to a design/spec pass before any runtime execution
- the user later said “All right, plesae promote the progress and conclusion so far into the `main` and closeup this branch/worktree.” -> once the study is done, the user wants the result promoted to the main checkout and the worktree cleaned up rather than left as a dangling research branch
- the user wanted a “serious research answer” and asked for a “decision-oriented conclusion” -> future similar mechanism studies should end with a crisp closure decision, not just logs or tentative notes

Key steps:

- read the CoordExp docs/router layer and historical diagnostic notes before designing the study, especially `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, `docs/eval/CONTRACT.md`, the duplication-collapse notes, and the raw-text mechanism notes
- used multiple read-only subagents to parallelize brainstorm lanes: experiment matrix, probe/statistics, causal patching, and local feasibility/sample-selection
- drafted and committed repo-local superpowers artifacts for the study: a design/spec memo and a worktree-ready implementation plan
- verified the fixed checkpoint directory exists and contains `coord_tokens.json`, so the study should be treated as a merged coord-token model surface rather than a raw-text digit-token surface
- used a worktree (`/data/CoordExp/.worktrees/qwen3-vl-instance-binding`) for the implementation slice and kept the main checkout clean until promotion
- discovered and fixed an idempotency bug in the merge stage: re-running the merge had accidentally re-ingested `*_merged.jsonl` and doubled row counts; the fix excluded prior merged files and added a regression test
- after the first pass, promoted the mechanism note from “temporary” to canonical progress status and then closed the loop after the core-diagnosis addendum
- merged the worktree branch back into `main`, then removed the worktree and deleted the branch

Reusable knowledge:

- the fixed checkpoint directory `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full` exists and contains `coord_tokens.json`
- the current repo route for this kind of mechanism study is a canonical note under `progress/diagnostics/`, not a benchmark note under `progress/benchmarks/`
- for this study shape, the `main` branch ended up ahead by a handful of commits after merge/closure, and the temp diagnostic note was renamed to a canonical findings note
- the branch/worktree cleanup pattern that worked here was: finish the research slice, merge it, rename/promote the progress note, then delete the worktree and the merged local branch

Failures and how to do differently:

- a naive merge pass initially would have doubled merged rows because the glob included `*_merged.jsonl`; the fix was to make merge aggregation ignore pre-existing merged outputs and add a regression test for that behavior
- `basedpyright` produced a large number of `Unknown`-type errors in the new research harness; this was not pursued into a broad typing-polish pass during closeout because runtime tests, ruff, YAML checks, and report regeneration were the important verification gates for the research slice
- there were stray untracked main-copy drafts of the spec/plan; they were backed up out of tree before merging so they would not interfere with promotion

Reusable knowledge:

- the result that was ultimately canonized was a mixed view: partial pre-`x1` binding exists, schema/pre-coordinate states act as a readout/carrier, and `x1/y1` remains the hard commitment boundary
- the strongest causal site in the final addendum was the bracket / immediate-pre-`x1` slot; desc-closing quote and field delimiter were nearly inert
- same-image same-desc donor patches transferred mass, but wrong-image same-syntax controls also disrupted target mass, so the schema/pre-coordinate effect is real but control-sensitive
- the worktree cleanup can safely remove the merged branch after promotion when the main checkout is clean and the merged commit is already on `main`

References:

- [1] promoted canonical findings note: `progress/diagnostics/2026-04-24_qwen3_vl_instance_binding_mechanism_findings.md`
- [2] router/index updates: `progress/diagnostics/README.md`, `progress/index.yaml`
- [3] merged closure commit on `main`: `c2077f4 docs(progress): close qwen3 vl binding study`
- [4] merge commit: `dc39493 merge: qwen3 vl instance binding study`
- [5] worktree cleanup target: `/data/CoordExp/.worktrees/qwen3-vl-instance-binding`
- [6] final conclusions file content included the consolidated mechanism statement: `converged_mixed_partial_pre_x1_binding_with_pre_coordinate_readout`
- [7] verification from `main`: `PYTHONPATH=. conda run -n ms python -m pytest tests/test_qwen3_vl_instance_binding_study.py -q` -> `23 passed`; `ruff check` -> `All checks passed`; YAML sanity for `progress/index.yaml` and the Qwen3-VL configs -> `YAML_OK`

