thread_id: 019dd846-2392-7431-9d4a-8758523ab1a7
updated_at: 2026-05-01T09:13:38+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/29/rollout-2026-04-29T08-06-08-019dd846-2392-7431-9d4a-8758523ab1a7.jsonl
cwd: /data/CoordExp
git_branch: main

# Focused Stage-1 ET-RMP-CE support-weight experiment was implemented, verified, and then diagnosed against eval artifacts.

Rollout context: The user asked for a narrow Stage-1 ET-RMP-CE objective variant in /data/CoordExp: decompose branch loss into valid-support and valid-balance terms, expose config weights, add metrics, keep decoding unchanged (including repetition_penalty=1.10), and avoid broader refactors. Later, the same thread was extended with a diagnosis request about whether the run’s behavior was really an ordering/position issue or a model/objective issue.

## Task 1: Add support-weighted ET-RMP-CE branch objective

Outcome: success

Preference signals:
- The user explicitly required a “focused experimental change” and said “Do not introduce” decoding changes, RL/replay, freezing, architecture changes, or visual/language prior subtraction -> future ET-RMP work should stay tightly scoped to objective/config/metrics, not decoding or architecture.
- The user asked to keep eval decoding unchanged “including repetition penalty 1.10” -> future ET-RMP experiments should preserve the eval contract exactly unless the user changes it.
- The user wanted the change exposed as config parameters and metrics, not a broad refactor -> future similar changes should prefer config-first knobs and metric surfacing.

Key steps:
- Read the repo’s Stage-1 ET-RMP docs/config/tests and found the current implementation points: `src/trainers/stage1_set_continuation/full_suffix.py`, `trainer.py`, `src/config/schema.py`, `src/trainers/stage1_set_continuation/metrics.py`, plus tests and `configs/stage1/set_continuation/rmp_ce.yaml`.
- Added TDD coverage first; the initial tests failed as expected because the branch-weight config and metric whitelist did not exist yet.
- Implemented the branch loss decomposition in `full_suffix.py`: explicit branch support loss, branch balance loss, branch total, and preserved branch CE for backward comparability.
- Added config plumbing through `Stage1SetContinuationObjectiveConfig` and trainer passthrough so `objective.branch_support_weight` and `objective.branch_balance_weight` are available and validated.
- Expanded ET-RMP metrics to include `loss/rmp_branch_support`, `loss/rmp_branch_balance`, `loss/rmp_branch_total`, and valid-child mass summaries (`min`, `p10`, `p50`, `p90`, and type buckets).
- Updated `configs/stage1/set_continuation/rmp_ce.yaml` to a support-weighted profile (`branch_support_weight: 2.0`, `branch_balance_weight: 1.0`) and renamed the artifact/run provenance so it does not collide with the earlier equal-weight run.
- Updated Stage-1 objective/metrics docs and the OpenSpec delta to describe the support/balance split and the new experimental profile.
- Re-ran the targeted Stage-1 tests until green.

Failures and how to do differently:
- The first red run failed in exactly the expected places: missing config keys, missing function parameters, and missing metric whitelist entries. That confirmed the test-first approach was correctly pinning the contract.
- A later test failure was only a tiny float mismatch between `math.log` and Torch’s `logsumexp`; switching the test to use the same Torch math resolved it without loosening the production implementation.

Reusable knowledge:
- The ET-RMP branch path lives in `src/trainers/stage1_set_continuation/full_suffix.py` and the trainer entrypoint is `_process_full_suffix_batch` in `src/trainers/stage1_set_continuation/trainer.py`.
- `Stage1SetContinuationObjectiveConfig` originally only had `mode` and `suffix_order`; adding ET-RMP-specific branch weights required a schema extension, not just a YAML tweak.
- `EMITTED_STAGE1_SET_CONTINUATION_METRICS` is the whitelist that determines whether new trainer metrics survive emission.
- The checked-in support-weight profile is named around support2 in artifacts/configs; the earlier equal-weight ET-RMP profile already existed and should be kept distinct for provenance.

References:
- [1] Tests used as the contract: `tests/test_stage1_set_continuation_full_suffix.py`, `tests/test_stage1_set_continuation_config.py`, `tests/test_stage1_set_continuation_metric_keys.py`, `tests/test_stage1_set_continuation_benchmark_profiles.py`, `tests/test_stage1_set_continuation_trainer_smoke.py`.
- [2] Branch-loss implementation surface: `src/trainers/stage1_set_continuation/full_suffix.py` (`_step_nll`, `compute_full_suffix_loss`, `score_full_suffix_retained`, `score_full_suffix_batch_retained`).
- [3] Config/metric surfaces: `src/config/schema.py`, `src/trainers/stage1_set_continuation/metrics.py`, `configs/stage1/set_continuation/rmp_ce.yaml`.
- [4] Documentation/spec updates: `docs/training/STAGE1_OBJECTIVE.md`, `docs/training/METRICS.md`, `docs/training/STAGE1_ET_RMP_CE.md`, `openspec/changes/add-stage1-et-rmp-ce-objective/specs/stage1-set-continuation-training/spec.md`.

## Task 2: Diagnose support-weight run vs baseline and test the ordering hypothesis

Outcome: partial

Preference signals:
- After implementation, the user asked to “continue,” and the later analysis focused on whether the remaining problem was really sorted-order traversal / late-object recall -> future similar follow-ups should start from artifacts and compare matched rollouts, not from pure metric intuition.
- The user wanted the analysis grounded in current ET-RMP infrastructure and current artifacts, not guesses -> future diagnosis should prioritize metrics, matches, and config provenance.

Key steps:
- Re-opened the relevant ET-RMP docs and code to verify the ordering and sampling contract.
- Confirmed the data contract still uses top-left sorting for `custom.object_ordering: sorted` in the dataset layer, while Stage-1 continuation uses randomized prefix order and randomized full-suffix order during training.
- Compared the support-weighted run against the earlier ET-RMP baseline using eval artifacts under `output_remote/.../eval_detection/step_*`.
- Measured AP / AP50 / prediction totals / precision / recall / F1 from `metrics.json` across steps.
- Measured `rmp/valid_child_mass_mean`, p50, and type-split mass from training logs; the support-weighted run did not produce a clean upward shift in valid-child mass.
- Computed FN rate by GT ordinal position and by high-GT-count buckets, plus matched-order inversion rate using `matches.jsonl`.
- Inspected concrete per-image examples for a crowded image where support-weighted decoding still skipped many GT objects and another where it improved recall but still produced non-monotonic matched order.

Findings:
- JSON stability stayed clean: invalid JSON and empty predictions remained 0 in the compared eval artifacts.
- The support-weighted run did not reliably raise the key training diagnostic `rmp/valid_child_mass_mean`; it fluctuated around the same ballpark as baseline rather than increasing substantially.
- Final support-weighted eval at step 916 had better recall than earlier support steps, but it did not beat the baseline cleanly on crowded/high-GT images; it also emitted fewer total predictions than baseline step 300.
- The biggest behavioral signal was position bias: later GT positions had worse FN rates, and the support-weighted run slightly improved early GT positions while degrading later ones.
- Matched prediction order was frequently non-monotonic, so the model is not following a stable top-left traversal policy under greedy decoding.
- This looks less like a broken sort implementation and more like a rollout/traversal bottleneck that support reweighting alone does not solve.

Failures and how to do differently:
- Serena symbol navigation could not resolve the repo paths in this session, so the analysis had to fall back to exact shell reads after narrowing with `rg`.
- An attempted config-path read initially looked in the wrong artifact path before being corrected.
- The analysis would be more reusable if future runs record these same diagnostics directly in eval artifacts (especially FN-by-ordinal and matched-order inversion rate), instead of reconstructing them ad hoc.

Reusable knowledge:
- The current training/eval contract for support-weighted ET-RMP is:
  - `custom.object_ordering: sorted`
  - `custom.object_field_order: desc_first`
  - `objective.mode: entry_trie_rmp_ce`
  - `objective.suffix_order: random`
  - `objective.branch_support_weight: 2.0`
  - `objective.branch_balance_weight: 1.0`
  - eval generation kept `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.1`.
- `support` did not collapse JSON validity; the remaining issue is recall/coverage, especially in crowded images and late GT positions.
- If recall is still weak after support reweighting, the next diagnostic should likely be about traversal/order/crowding or decode-time state mismatch, not just more branch weight.

References:
- [1] Comparison artifacts:
  - support run: `output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_support2_effbsz128_v1/.../v0-20260429-162104/eval_detection/step_0000916/metrics.json`
  - baseline run: `output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_v1/.../v0-20260429-022918/eval_detection/step_0000300/metrics.json`
- [2] Key metric deltas observed from those artifacts:
  - support final: `bbox_AP=0.418114`, `bbox_AP50=0.552477`, `f1ish@0.50_pred_total=962`, `precision_full_micro=0.844587`, `recall_full_micro=0.545706`, invalid/empty JSON = 0.
  - baseline step300: `bbox_AP=0.420454`, `bbox_AP50=0.562275`, `f1ish@0.50_pred_total=992`, `precision_full_micro=0.816475`, `recall_full_micro=0.542244`, invalid/empty JSON = 0.
- [3] Training log evidence from support run: `rmp/valid_child_mass_mean` hovered around `0.26-0.33`, with type mass dominated by `desc_text` and very low `coord` mass (around `0.02-0.03`).
- [4] Code evidence for the ordering contract: `src/datasets/utils.py:86` (`sort_objects_by_topleft`), `src/datasets/dense_caption.py:413` (`custom.object_ordering='sorted'` checks top-left order), `src/trainers/stage1_set_continuation/sampling.py:83` (`prefix_order`), `src/trainers/stage1_set_continuation/full_suffix.py:146` (`suffix_order`).
- [5] Concrete crowded-image examples from `matches.jsonl` showed the support run could still skip many objects in top-left order, especially in images with 11+ GT objects.
