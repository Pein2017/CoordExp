---
phase: 05-validation-and-best-k-readiness
plan: 03
status: completed
subsystem: best-k-readiness
tags: [stage2, k4, k2, pseudo_positive, ablation]
requires:
  - 05-01
  - 05-02
provides:
  - Enabled `K=4` vs enabled `K=2` runtime comparison
  - Runtime proof that enabled `K=2` remains a no-promotion control
affects: [phase-05]
tech-stack:
  added:
    - configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_k2_4steps.yaml
  patterns:
    - Same smoke surface, different `num_rollouts`
key-files:
  created:
    - configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_k2_4steps.yaml
  modified: []
requirements-completed: [VAL-03, VAL-04]
completed: 2026-03-22
verification:
  - "`server_gpus=2 train_gpus=3,4,5,6 config=configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_k2_4steps.yaml wait_timeout=120 conda run -n ms bash scripts/train_stage2.sh`"
---

# Phase 5 Plan 03 Summary

**Completed: best-`K` runtime comparison is now evidenced and the enabled `K=2` control behaves correctly**

## Accomplishments
- Created a committed enabled `K=2` smoke variant that extends the exact `K=4`
  smoke surface and changes only:
  - `stage2_ab.channel_b.triage_posterior.num_rollouts`
  - rollout-server port
  - run/output names
- Ran the enabled `K=2` control through the same combined launcher topology.
- Compared final-step rate-based triage metrics between enabled `K=4` and
  enabled `K=2`.

## Runtime Comparison
- Enabled `K=4` final step:
  - `pseudo_positive_selected_count=2.0`
  - `pseudo_positive_selected_support_rate_num/den=5.0/6.0`
- Enabled `K=2` final step:
  - `pseudo_positive_selected_count=0.0`
  - `pseudo_positive_selected_support_rate_num/den=0.0/0.0`
- Both runs:
  - `stage2/channel_b=1.0`
  - `stage2/raw_rollouts=8.0`
  - `stage2_ab/channel_b/dup/N_duplicate_bursts=0.0`
  - `train/optimization/loss_dead_anchor_suppression=0.0`

## Qualitative Review
- `K=4` monitor dump showed a captured sample with:
  - `valid_explorer_count=3`
  - two pseudo-positive winners
  - zero duplicate bursts
  - atomic `clock` / `vase` predictions rather than an oversized group box
- `K=2` monitor dump showed:
  - `valid_explorer_count=1`
  - no pseudo-positive winners
  - one shielded anchor and one dead anchor
  - zero duplicate bursts

## Conclusion
- The enabled `K=4` default profile is live and measurable.
- The enabled `K=2` path is a true runtime no-promotion control, not just a unit
  test invariant.
- The worktree is ready for the next `best-K` ablation stage without widening
  pseudo-positive semantics further.

---
*Phase: 05-validation-and-best-k-readiness*
*Completed: 2026-03-22*
